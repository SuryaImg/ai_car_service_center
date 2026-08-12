import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from livekit.agents import RunContext, function_tool

# ---------------------------------------------------------------------------
# Timezone helpers — all call timestamps/dates use IST, never UTC.
# ---------------------------------------------------------------------------

IST = timezone(timedelta(hours=5, minutes=30))

DEFAULT_CALL_TYPE = "service"
UNIFIED_AGENT_NAME = "car-service-assistant"

SERVICE_CENTER_NAME = "Škoda Service Center"


def now_ist() -> datetime:
    """Current time in Indian Standard Time (UTC+5:30)."""
    return datetime.now(IST)


def today_ist_str() -> str:
    """Today's date in IST, formatted YYYY-MM-DD."""
    return now_ist().strftime("%Y-%m-%d")


@dataclass
class CallState:
    user_id: str
    customer_record: Optional[dict] = None
    vehicle_record: Optional[dict] = None
    vehicle_brand_blocked: bool = False
    blocked_vehicle_label: Optional[str] = None
    pending_vehicle_details: dict = field(default_factory=dict)
    booking_draft: Optional[dict] = None
    service_type: Optional[str] = None
    outcome: str = "unresolved"
    notes: list = field(default_factory=list)
    call_history_entry: Optional[dict] = None
    call_type: str = "service"
    is_outbound: bool = False
    outbound_purpose: str = ""
    forced_language: Optional[str] = None
    current_language: Optional[str] = None
    phase: str = "ACTIVE"
    unclear_turn_streak: int = 0
    _last_ask_anything_else_turn: int = -10
    identity_mismatch: bool = False
    turn_seq: int = 0

    def log(self, event: str) -> None:
        self.notes.append(f"[{now_ist().isoformat()}] {event}")


PHASE_ACTIVE = "ACTIVE"
PHASE_AWAITING_CONFIRMATION = "AWAITING_CONFIRMATION"
PHASE_CLOSED = "CLOSED"

_CALL_STATES: dict[str, CallState] = {}


def get_call_state(user_id: str) -> CallState:
    if user_id not in _CALL_STATES:
        _CALL_STATES[user_id] = CallState(user_id=user_id)
    return _CALL_STATES[user_id]


def clear_call_state(user_id: str) -> None:
    """
    Remove a finished call's state from memory. Must be called when a call/session
    ends so that _CALL_STATES does not grow unboundedly (memory leak) across the
    life of the worker process.
    """
    _CALL_STATES.pop(user_id, None)


def cancel_stale_confirmation(state: "CallState") -> None:
    """
    FIX (issue 8A — premature call-ending on a stale confirmation): if a
    prior ask_anything_else confirmation is still pending (phase ==
    AWAITING_CONFIRMATION) and the customer has now engaged with a
    genuinely different topic — evidenced by some OTHER tool being called —
    that confirmation prompt is stale. Cancel it so a later, unrelated
    yes/no from the customer can never be mistaken for the answer to it.
    Called from every substantive tool in this file and in services.py,
    except ask_anything_else/record_call_outcome themselves.
    """
    if state.phase == PHASE_AWAITING_CONFIRMATION:
        state.phase = PHASE_ACTIVE
        state.log("New topic engaged while awaiting confirmation — confirmation wait cancelled.")


PERSONALITY_PROMPT = """
AGENT PERSONA & GENDER RULES — ABSOLUTE, NON-NEGOTIABLE
You are LILY, a female Service Advisor representing Škoda Service Center. Use feminine Hindi/Hinglish verb forms ONLY.
RULE: any verb ending in "-ta hoon/hu/hai", "-sakta hoon", "-raha hoon" is masculine and FORBIDDEN — use "-ti hoon" / "-sakti hoon" / "-rahi hoon" instead.
Examples: "Main check karti hoon" (not "karta hoon"); "Main book kar sakti hoon" (not "kar sakta hoon"); "Main dekh rahi hoon" (not "dekh raha hoon").
Before every Hindi/Hinglish reply, scan for masculine endings and fix them. No exceptions — not for empathy lines, not ever.
VOICE STYLE:
- Speak like a friendly, professional service advisor at an authorized car service center.
- Maintain natural Indian English pronunciation.
- Do not use an American accent.
- Keep Hinglish pronunciation natural for Indian customers.
Tone: professional, warm, knowledgeable, calm — an experienced human service advisor, not a bot.
"""

GUARDRAILS_PROMPT = """
CONTEXT ONLY & HALLUCINATION PREVENTION
- The loaded dealer database JSON (vehicles, customers, catalog, slots) is the ONLY source of truth. Never invent slot times, service prices, vehicle specs, or center policies.
- Never assume a detail applies just because it's typical — before answering a specific factual question, call the relevant tool for THIS record rather than recalling from memory or general knowledge.
- If information is missing: "Mere paas is samay uski exact jaankari available nahi hai." Ask a clarifying question or explain the limitation, then continue helping. Never move toward closing the call because you don't know the answer.
NO COMMITMENT WITHOUT A TOOL CALL
- Never verbally promise to book a slot, apply discounts, or act on ANY request unless you actually call the matching tool for it in that same turn.
- If no tool exists for what's being asked, do NOT agree to it — decline politely and redirect to what you can actually help with.

CUSTOMER DATA PRIVACY & ACCURACY — STRICTLY ENFORCED
- Never disclose another customer's or vehicle's data. Each call is scoped to the verified vehicle/customer loaded this session.
- The loaded vehicle/customer record is the ONLY source of truth. Never guess or invent details.
- If a field or value is unavailable in the record, explicitly say that you do not have that information instead of making one up.
"""

LANGUAGE_RULES_PROMPT = """
LANGUAGE DETECTION & MIRRORING — HIGHEST PRIORITY, EVERY SINGLE TURN
This is a live voice call. You must decide the reply language yourself, fresh, on
EVERY turn, based on the actual words/audio the caller JUST spoke — do not wait to be
told, do not rely on any earlier turn's language, and do not carry a language forward
just because you used it before.
  * Caller's latest message is in English → reply in English only.
  * Caller's latest message is in Hindi/Devanagari → reply in Hindi/Hinglish.
  * Caller's latest message mixes both → reply in Hinglish.
A "CURRENT LANGUAGE" block may also appear below, reflecting a background best-effort
estimate — treat it only as a hint, never as an override of what you actually just
heard the caller say. If they conflict, always trust what the caller just said over
the block.
Never use shuddh/overly formal Hindi. Never use masculine verb forms (see Gender Rules).
OVERRIDE: if the caller explicitly says "speak in English" / "English mein baat karo",
switch and stay in English for the rest of the call regardless of what language they
use afterward.
"""

INTERRUPTION_PROMPT = """
INTERRUPTION & QUESTION PRIORITY RULES
- Let the customer finish speaking before responding — a brief mid-thought pause is normal, not the end of their turn.
- If they interrupt you, stop immediately and let them finish.
- After they finish, check what they actually said before deciding how to respond:
  * If it was just a short acknowledgment/filler ("haan", "ji", "hmm", "ok", "achha", "theek hai") with no real question or new statement — they are listening, not taking the floor. RESUME your unfinished sentence from exactly where you stopped (same words/meaning) and complete the thought, then continue naturally. Do NOT skip ahead to a different question, and do NOT treat the filler as an answer to something you hadn't asked yet.
  * If it was an actual question or new statement, address that first before returning to anything you were mid-way through.
- Always address the latest question first; don't switch topics before fully answering it.
- If after careful reading you are STILL unsure what the customer is asking, do NOT guess or answer incorrectly. Instead, ask a short clarifying question: "Maaf kijiye, kya aap thoda aur bata sakte hain — aap [X] ke baare mein pooch rahe hain ya [Y] ke baare mein?"
CLARIFICATION RULE:
- Any clarification question MUST follow the current conversation language.
- Never force Hindi if the caller is speaking English.
- Never force English if the caller is speaking Hindi/Hinglish.
- Asking a clarifying question is the required response whenever you are unsure.
- Never call ask_anything_else or record_call_outcome because of unclear audio, uncertainty, or an incomplete request. Continue the conversation until the caller's request is understood or they explicitly choose to end the call.
"""

_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
_ALPHA_RE = re.compile(r"[^\W\d_]", re.UNICODE)

_HINGLISH_WORDS = {
    "hai", "hain", "ka", "ki", "ke", "kya", "kaise", "kab", "kahan", "kyun",
    "kyu", "nahi", "nahin", "aap", "aapka", "aapki", "aapke", "mujhe", "mera",
    "meri", "mere", "hoon", "hu", "raha", "rahi", "rahe", "karo", "karna",
    "kar", "diya", "diye", "chahiye", "gaadi", "car", "service", "slot",
    "abhi", "thoda", "bhi", "toh", "to", "bas", "theek", "thik", "achha",
    "acha", "haan", "han", "ji", "kripya", "dhanyavaad", "shukriya", "madad",
    "jaankari", "batao", "bataye", "batayein", "hoti", "hogi", "hota", "hote",
    "kitni", "kitna", "kitne", "wala", "wali", "wale", "liye", "saath",
    "phir", "fir", "kal", "aaj", "din", "baat", "sab", "sabhi",
    "koi", "kuch", "yeh", "ye", "voh", "vo", "uska", "uski", "unka",
}

_ENGLISH_LOCK_PHRASES = (
    "speak in english", "speak english", "reply in english", "respond in english",
    "answer in english", "talk in english", "english mein baat karo",
    "english mein bolo", "english mein baat kijiye", "please english",
    "can you speak english", "switch to english",
)
_SWITCH_CONFIDENCE = 0.5


def wants_english_lock(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return any(phrase in lowered for phrase in _ENGLISH_LOCK_PHRASES)


def _classify_utterance(text: str) -> tuple[str, float]:
    if not text or not text.strip():
        return ("hinglish", 0.0)

    devanagari_chars = len(_DEVANAGARI_RE.findall(text))
    if devanagari_chars:
        alpha_chars = len(_ALPHA_RE.findall(text)) or 1
        devanagari_ratio = devanagari_chars / alpha_chars
        if devanagari_ratio >= 0.5:
            return ("hindi", min(1.0, 0.6 + devanagari_ratio))
        return ("hinglish", 0.5)

    words = re.findall(r"[a-zA-Z']+", text.lower())
    if not words:
        return ("hinglish", 0.0)

    length_factor = min(1.0, len(words) / 3.0)
    hinglish_hits = sum(1 for w in words if w in _HINGLISH_WORDS)
    ratio = hinglish_hits / len(words)

    if hinglish_hits == 0:
        return ("english", 0.9 * length_factor)
    if ratio >= 0.6:
        return ("hinglish", 0.85 * length_factor)
    if ratio >= 0.25:
        return ("hinglish", 0.55 * length_factor)

    return ("hinglish", 0.3 * length_factor)


def resolve_current_language(call_state, text: str) -> str:
    if getattr(call_state, "forced_language", None) == "english":
        call_state.current_language = "english"
        return "english"

    if wants_english_lock(text):
        call_state.forced_language = "english"
        call_state.current_language = "english"
        return "english"

    label, confidence = _classify_utterance(text)
    current = getattr(call_state, "current_language", None)
    if current is None or confidence >= _SWITCH_CONFIDENCE or label == current:
        call_state.current_language = label
    return call_state.current_language


_LANGUAGE_TRAILER_BODY = {
    "english": "Current language: English. Reply only in English. Use natural Indian-English pronunciation, rhythm, and intonation throughout — never switch to an American or British accent mid-response. Sound like a female Indian service advisor at Škoda. Ignore previous conversation languages.",
    "hindi": "Current language: Hindi. Reply only in natural spoken Hindi/Hinglish. Use Indian pronunciation and feminine Hindi forms. Ignore previous conversation languages.",
    "hinglish": "Current language: Hinglish. Reply naturally in Hinglish (mixed Hindi-English), matching the caller's own mix with an Indian speaking style. Ignore previous conversation languages.",
}


def language_instruction_trailer(lang: str) -> str:
    body = _LANGUAGE_TRAILER_BODY.get(lang, _LANGUAGE_TRAILER_BODY["hinglish"])
    return "\n\n=========================================\nCURRENT LANGUAGE (auto-detected locally, re-evaluated every turn)\n=========================================\n" + body


CONVERSATION_RULES_PROMPT = """=========================================
CONVERSATION & ACTIVE LISTENING RULES
=========================================
- Vary listening cues naturally ("Ji boliye.", "Bilkul.", "Samajh gayi.", "Haan ji.") — never repeat back-to-back. Sound like an experienced human service advisor, not a script.
- Retain everything said this call (name, vehicle number, service requirement, preferred date/time) — never re-ask for it.
- If the customer asks you to recall something from earlier in THIS call, answer strictly from what was actually said in this conversation so far — never invent or guess.
- Stay consistent — don't contradict a detail you already quoted.
- If intent shifts mid-call, acknowledge it naturally and move on.
- SELF-CONSISTENCY: if your own last message asked a specific question, and the customer's reply is garbled, unclear, or off-topic, do NOT silently move on to a different, generic question (e.g. "kya main kisi aur tarah se madad kar sakti hoon?"). Either ask them to repeat their answer to the SAME question you just asked, or re-ask that exact question in slightly different words — never substitute a different question and act as if that's what you'd asked.

DIRECT ANSWER RULE:
- Answer follow-ups directly with only what was asked — no recapping prior answers ("Maine abhi bataya tha..." is forbidden). Treat every follow-up as a fresh, self-contained question.

ACKNOWLEDGEMENT RULE:
- A request should be acknowledged only once. After acknowledging it, treat it as completed. Do NOT repeat "aapki booking note kar li gayi hai" or "dhanyavaad" repeatedly.
- Respond directly to what the customer says in their latest message.
"""

OUT_OF_SCOPE_PROMPT = """
OUT-OF-SCOPE TOPIC HANDLING — STRICTLY ENFORCED
- You are a Car Service Assistant for Škoda Service Center. You assist with service booking, slot availability, service catalog/pricing, and vehicle lookup.
- If the caller asks anything completely unrelated (e.g., buying new cars, insurance claims, dealership ownership, non-automotive topics), politely state that you can only help with car service bookings and workshop queries for this call.
- Example: "Maaf kijiye, main sirf car service booking, slot availability, aur service packages ke sambandh mein madad kar sakti hoon."
- Never partially engage with out-of-scope topics. Politely redirect back to car service.
"""

CALLER_GENDER_NEUTRAL_PROMPT = """=========================================
CALLER GENDER — ALWAYS NEUTRAL, NON-NEGOTIABLE
You don't know the caller's gender — never assume male or female. Use gender-neutral forms for THEM (feminine stays reserved for yourself, Lily).
Examples: "Kya aap apna naam bata sakte hain?" (not "sakti hain"); "Aap kab aana chahenge?" (not "chahengi").
RULE: yourself → always feminine. The caller → always neutral. Never mix these up.
"""

CALL_SUMMARY_PROMPT = """
CALL CLOSURE:
- Never end a call immediately after answering one question.
- Continue helping naturally if the caller has more questions.
- NEVER say "kya aap kuch aur jaanna/puchna chahte hain?", "anything else?", or any equivalent phrase as a generic habit after a normal answer. This exact question is reserved EXCLUSIVELY for the closing sequence below and may only be asked in the SAME turn you call the ask_anything_else tool.

Before ending any call:
1. Call ask_anything_else and wait for the caller's reply.
2. If the caller clearly says they have no more questions, call ask_anything_else one more time and wait for the reply.
3. If the caller again clearly confirms they have no more questions, call record_call_outcome, give a brief summary if needed, and end the call politely.
4. Otherwise, continue helping the caller normally.

IMPORTANT:
- Any new question, request, or concern from the caller at any point immediately cancels the closing process.
- "Thank you", "Okay", "Haan", "Theek hai" alone do NOT mean the caller wants to end the call.
- Never say goodbye or use any closing line unless record_call_outcome explicitly allows the call to end.
"""

COMMON_PROMPT = (
    PERSONALITY_PROMPT + "\n" +
    GUARDRAILS_PROMPT + "\n" +
    OUT_OF_SCOPE_PROMPT + "\n" +
    LANGUAGE_RULES_PROMPT + "\n" +
    INTERRUPTION_PROMPT + "\n" +
    CONVERSATION_RULES_PROMPT + "\n" +
    CALLER_GENDER_NEUTRAL_PROMPT + "\n" +
    CALL_SUMMARY_PROMPT
)


def build_prompt(domain_prompt: str) -> str:
    """Combines the shared COMMON_PROMPT with domain-specific guidelines."""
    return COMMON_PROMPT + "\n" + domain_prompt


logger = logging.getLogger("common")

INTENT_CATALOG: dict[str, dict] = {
    "greeting": {
        "category": "Greeting",
        "examples": [
            "Hello", "Namaste", "Hi", "Good morning", "Haan ji", "Is this Skoda service center?",
        ],
    },
    "book_service": {
        "category": "Book Service Request",
        "examples": [
            "I want to book a service appointment.", "Mujhe gaadi ki service karwani hai.",
            "Service booking karni hai.", "I need to schedule a service.",
        ],
    },
    "check_slot": {
        "category": "Check Available Slots",
        "examples": [
            "What slots are available tomorrow?", "Kal ka time slot hai kya?",
            "Available time slots batao.", "When is the next available slot?",
        ],
    },
    "periodic_maintenance": {
        "category": "Periodic Maintenance",
        "examples": [
            "I need basic service.", "Standard service karwani hai.",
            "Major service package details?", "Periodic maintenance charge kya hai?",
        ],
    },
    "repair_request": {
        "category": "Repair Service",
        "examples": [
            "Brake issue hai.", "Suspension check karwana hai.",
            "Engine diagnostics chahiye.", "Clutch problem hai.",
        ],
    },
    "electrical_ac": {
        "category": "Electrical & AC Service",
        "examples": [
            "AC cooling nahi kar raha.", "Battery check karwani hai.",
            "Headlight not working.", "AC service rate kya hai?",
        ],
    },
    "detailing_request": {
        "category": "Detailing & Washing",
        "examples": [
            "Car wash karwani hai.", "Deep cleaning costs?",
            "Ceramic coating available hai?", "Denting painting charges?",
        ],
    },
    "symptom_report": {
        "category": "Symptom Report",
        "examples": [
            "Awaaz aa rahi hai engine se.", "Brake me squeaking sound hai.",
            "Car starting problem hai.", "Mileage kam de rahi hai.",
        ],
    },
    "cost_inquiry": {
        "category": "Cost / Price Inquiry",
        "examples": [
            "Kitna kharcha aayega?", "Service price kya hai?",
            "Total cost kitna hoga?", "Charges batao.",
        ],
    },
    "complaint": {
        "category": "Customer Complaint",
        "examples": [
            "Pehli service theek se nahi hui.", "Price bahut zyada hai.",
            "Service quality poor hai.", "Manager se baat karwao.",
        ],
    },
    "anger_mild": {
        "category": "Customer Anger - Mild",
        "examples": [
            "Kitna time laga rahe ho?", "Bar bar poochna padta hai.",
        ],
    },
    "anger_strong": {
        "category": "Customer Anger - Strong",
        "examples": [
            "Aapki service bilkul kharab hai.", "Complaint karunga main.",
        ],
    },
    "human_transfer": {
        "category": "Human Transfer Request",
        "examples": [
            "Service manager se baat karwao.", "Human agent chahiye.",
            "Transfer call to manager.",
        ],
    },
    "call_ending": {
        "category": "Call Ending",
        "examples": [
            "That's all thank you.", "No more questions.",
            "Bas itna hi tha, dhanyavaad.", "Okay bye.",
        ],
    },
}

RESPONSE_GUIDANCE: dict[str, str] = {
    "greeting": (
        "Greet the customer warmly as Lily from Škoda Service Center. "
        "Ask how you can assist them today with service booking, slot availability, or service pricing."
    ),
    "book_service": (
        "Ask for their Vehicle Registration Number or Registered Mobile Number to look up their record in the database."
    ),
    "check_slot": (
        "Use get_available_slots to list open time slots for the requested date and present options clearly."
    ),
    "periodic_maintenance": (
        "Use get_service_catalog to share details and pricing for Basic, Standard, or Major service."
    ),
    "repair_request": (
        "Acknowledge the repair requirement, share details from service catalog, and offer slot booking for inspection."
    ),
    "electrical_ac": (
        "Share AC/battery service package options from get_service_catalog and check available slots."
    ),
    "detailing_request": (
        "Share car wash, deep cleaning, or ceramic coating details from get_service_catalog."
    ),
    "symptom_report": (
        "Use get_symptom_recommendation to recommend the appropriate service and offer slot booking."
    ),
    "cost_inquiry": (
        "Share transparent price estimates from get_service_catalog and explain what's included."
    ),
    "complaint": (
        "Respond with empathy, reassure the customer, and offer to log their feedback or escalate to the Service Manager via escalate_to_human."
    ),
    "anger_mild": (
        "Stay calm, polite, and attentive. Apologize for any inconvenience and prioritize answering their request."
    ),
    "anger_strong": (
        "Remain calm and professional. Acknowledge their concern with empathy and offer immediate escalation to the Service Manager."
    ),
    "human_transfer": (
        "Use escalate_to_human tool. Inform the customer that a senior Service Advisor / Manager will contact them."
    ),
    "call_ending": (
        "First ask if they need help with anything else. If confirmed no, summarize the booking details and end politely."
    ),
}


@dataclass
class IntentLog:
    """Tracks every classified intent during a call, for analytics/QA."""
    entries: list = field(default_factory=list)

    def add(self, intent_id: Optional[str], raw_utterance: str) -> None:
        self.entries.append(
            {
                "timestamp": now_ist().isoformat(),
                "intent_id": intent_id,
                "category": INTENT_CATALOG.get(intent_id, {}).get("category", "unclassified"),
                "utterance": raw_utterance,
            }
        )

    def already_logged(self, intent_id: Optional[str]) -> bool:
        return bool(intent_id) and any(e["intent_id"] == intent_id for e in self.entries)


@function_tool
async def record_customer_intent(ctx: RunContext, utterance: str, intent_id: str = "") -> str:
    """
    Call this whenever the customer expresses a clear intent (booking request, slot check,
    repair query, symptom report, complaint, call ending).
    """
    intent_log: IntentLog = ctx.userdata.setdefault("intent_log", IntentLog())
    already_said = intent_log.already_logged(intent_id)
    intent_log.add(intent_id or None, utterance)

    state = ctx.userdata.get("call_state")
    if state is not None and intent_id != "call_ending":
        cancel_stale_confirmation(state)

    if already_said:
        return (
            "You already acknowledged this earlier in the call — respond directly "
            "to what the customer is saying right now without repeating past lines."
        )

    guidance = RESPONSE_GUIDANCE.get(intent_id)
    if not guidance:
        return (
            "Acknowledge the customer's request calmly and professionally. "
            "Use available tools to provide accurate details from the service database."
        )

    logger.info("Customer intent recorded: %s | utterance=%r", intent_id, utterance)
    return guidance


@function_tool
async def handle_objection(ctx: RunContext, objection_type: str, details: str = "") -> str:
    """
    Call this for pushback or complaints regarding service cost, slot unavailability,
    or service quality. Returns calm de-escalation guidance.
    """
    guidance = RESPONSE_GUIDANCE.get(
        objection_type,
        "Stay calm, acknowledge their concern with empathy, explain standard service center policies, "
        "and offer escalation to the Service Manager if unresolved.",
    )
    state = ctx.userdata.get("call_state")
    if state is not None:
        cancel_stale_confirmation(state)
        state.log(f"Objection handled: {objection_type} | {details}")
    return guidance


@function_tool
async def switch_language(ctx: RunContext, requested_language: str) -> str:
    """
    Call this when the caller explicitly asks to continue in another language.
    Supports Hindi, Hinglish, and English.
    """
    lang = requested_language.strip().lower()
    supported = {"hindi", "english", "hinglish"}
    state = ctx.userdata.get("call_state")
    if state is not None:
        cancel_stale_confirmation(state)
        state.log(f"Language switch requested: {requested_language}")

    if lang in supported:
        return f"Switch fluidly to {requested_language}, keeping the same calm, respectful, professional tone."
    return (
        f"You are fluent in English, Hindi, and Hinglish. "
        "Apologize politely and explain that you can continue in Hindi or English."
    )


@function_tool
async def escalate_to_human(ctx: RunContext, reason: str) -> str:
    """
    Call this when the customer explicitly asks to speak with a human agent,
    the Service Manager, or has a complex grievance.
    """
    state = ctx.userdata.get("call_state")
    if state is not None:
        cancel_stale_confirmation(state)
        state.log(f"Escalation requested: {reason}")
        state.outcome = "escalated"

    return (
        "Confirm to the customer that you have noted their request and that a senior Service Advisor / "
        "Service Manager will call them back shortly on their registered number. "
        "Thank them for their patience and continue helping with any other questions."
    )


_QUESTION_WORDS = ("kya", "kaun", "kab", "kitna", "kitni", "kitne", "kaise",
                   "kahan", "kyun", "kyu", "what", "how", "when", "why",
                   "where", "which", "who")
_AFFIRM_DENY_WORDS = {
    "yes", "no", "yeah", "yep", "nope", "haan", "han", "nahi", "nahin",
    "theek", "thik", "ok", "okay", "bilkul", "sahi", "correct", "right",
    "achha", "acha", "sure", "done", "confirmed",
}


def _looks_like_affirmation_or_denial(text: str) -> bool:
    if not text:
        return False
    words = set(re.findall(r"[a-zA-Z\u0900-\u097F']+", text.lower()))
    return bool(words & _AFFIRM_DENY_WORDS)


@function_tool
async def ask_anything_else(ctx: RunContext) -> str:
    """
    Call this ONLY when you have already genuinely understood and fully
    responded to the caller's last message, and are now naturally moving on
    to check if they need anything more — this is the required first step
    before a call can be closed.
    """
    state: CallState = ctx.userdata["call_state"]
    now_turn = state.turn_seq
    if now_turn - state._last_ask_anything_else_turn < 2:
        return (
            "You just asked this very recently — asking it again this soon "
            "looks repetitive to the caller. Only call this again after "
            "genuinely answering something new."
        )
    state._last_ask_anything_else_turn = now_turn
    state.phase = PHASE_AWAITING_CONFIRMATION
    return (
        "Noted — now ask the caller if there's anything else they need help "
        "with, in your own natural words. Do NOT say a closing/goodbye line "
        "yet; that is only allowed after they respond and you call "
        "record_call_outcome."
    )


@function_tool
async def record_call_outcome(ctx: RunContext, outcome: str, caller_reply: str, summary: str = "") -> str:
    """
    Call this once, near the end of every call, to record the final outcome and a short summary.
    caller_reply MUST be the caller's actual last message — this is validated to make sure it's a real
    confirmation, not a question or unrelated remark, before closing is allowed.
    """
    state: CallState = ctx.userdata["call_state"]
    state.outcome = outcome or state.outcome
    state.log(f"Call outcome recorded: {outcome} | {summary}")
    state.call_history_entry = {
        "timestamp": now_ist().isoformat(),
        "call_type": ctx.userdata.get("call_type", "service"),
        "outcome": state.outcome,
        "summary": summary,
        "notes": list(state.notes),
    }

    # FIX (issue 8A — premature call-ending on a stale confirmation): phase
    # only ever flips to AWAITING_CONFIRMATION inside ask_anything_else, and
    # previously only flipped back inside this function's own rejection
    # branch below. If the customer asked unrelated new questions (calling
    # other tools) after that ask_anything_else and BEFORE this call, the
    # phase was still stuck at AWAITING_CONFIRMATION — so a later, unrelated
    # "no"/"नहीं" would incorrectly satisfy the confirmation check and close
    # the call. Require the confirmation to be answered on the very next
    # customer turn; otherwise treat it as stale and withhold closing.
    turns_since_ask = state.turn_seq - state._last_ask_anything_else_turn
    if outcome != "wrong_person" and turns_since_ask > 1:
        state.phase = PHASE_ACTIVE
        state.log(
            f"record_call_outcome: confirmation is stale ({turns_since_ask} turns since "
            "ask_anything_else) — closing withheld."
        )
        return (
            "The 'anything else?' confirmation is stale — the caller has asked other things since "
            "then, so this reply does not answer that question anymore. Do NOT close the call. If "
            "the caller is truly done now, call ask_anything_else again fresh and wait for their "
            "direct reply to THAT before calling record_call_outcome."
        )

    if outcome != "wrong_person" and state.phase != PHASE_AWAITING_CONFIRMATION:
        state.log("record_call_outcome called before ask_anything_else confirmation — closing withheld.")
        return (
            "Outcome recorded, but you have NOT yet asked the caller if "
            "there's anything else, or they haven't had a chance to confirm "
            "yet. Do NOT say a closing/goodbye line. Continue the "
            "conversation normally and keep helping them."
        )

    stripped = (caller_reply or "").strip()
    first_word = stripped.lower().split()[0] if stripped.split() else ""
    if (
        not stripped
        or stripped.endswith("?")
        or len(stripped.split()) >= 6
        or first_word in _QUESTION_WORDS
        or not _looks_like_affirmation_or_denial(stripped)
    ):
        state.phase = PHASE_ACTIVE
        state.log("record_call_outcome: caller_reply was not a real confirmation — closing cancelled.")
        return (
            "That reply was not a clear 'no more questions' confirmation — "
            "it looks like a question, a repeat request, or something "
            "unrelated. Do NOT close the call. Answer what they actually "
            "said, then only return to ask_anything_else later if truly done."
        )

    state.phase = PHASE_CLOSED
    return (
        "Outcome recorded. Give ONE brief closing message only. "
        "If the caller speaks again before the call disconnects, "
        "immediately continue the conversation normally and do not repeat the previous closing."
    )