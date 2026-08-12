"""
triage_agent.py — The single LiveKit worker for Škoda Car Service Center Voice Assistant.

Handles inbound service booking calls, slot availability inquiries, repair/maintenance guidance,
and outbound periodic maintenance reminders.
"""

import asyncio
import json
import logging
import os
import re
from typing import Optional

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import (
    Agent,
    AgentSession,
    ChatContext,
    RoomInputOptions,
    RunContext,
    WorkerOptions,
    function_tool,
)
from livekit.plugins.google.realtime import RealtimeModel
from google.genai import types as genai_types
load_dotenv()
logger = logging.getLogger("service-agent")

try:
    from livekit.plugins import noise_cancellation
    _ROOM_INPUT_OPTIONS = RoomInputOptions(noise_cancellation=noise_cancellation.BVCTelephony())
except ImportError:
    logger.warning("noise_cancellation plugin not installed; running without BVCTelephony noise cancellation.")
    _ROOM_INPUT_OPTIONS = RoomInputOptions()

from common import (
    build_prompt,
    get_call_state,
    clear_call_state,
    CallState,
    record_customer_intent,
    handle_objection,
    switch_language,
    escalate_to_human,
    ask_anything_else,
    record_call_outcome,
    now_ist,
    today_ist_str,
    resolve_current_language,
    language_instruction_trailer,
    UNIFIED_AGENT_NAME,
    SERVICE_CENTER_NAME,
)
from livekit.agents.llm import ChatMessage, StopResponse
from services import (
    load_all_car_service_data,
    find_vehicle_by_registration,
    find_customer_by_phone,
    find_customer_by_id,
    lookup_vehicle_by_registration,
    lookup_customer_by_phone,
    collect_unregistered_vehicle_details,
    check_customer_profile,
    update_customer_profile,
    get_service_catalog,
    get_symptom_recommendation,
    get_available_slots,
    create_service_booking,
    generate_booking_summary,
    send_booking_confirmation,
    persist_booking_call_history,
    mark_service_reminder_sent,
)

_REALTIME_INPUT_CONFIG = genai_types.RealtimeInputConfig(
    automatic_activity_detection=genai_types.AutomaticActivityDetection(
        disabled=False,
        start_of_speech_sensitivity=genai_types.StartSensitivity.START_SENSITIVITY_LOW,
        end_of_speech_sensitivity=genai_types.EndSensitivity.END_SENSITIVITY_LOW,
        prefix_padding_ms=300,
        silence_duration_ms=700,
    )
)

_CONTEXT_PRESERVATION_KWARGS = dict(
    context_window_compression=genai_types.ContextWindowCompressionConfig(
        sliding_window=genai_types.SlidingWindow(),
    ),
    session_resumption=genai_types.SessionResumptionConfig(),
)

_FALSE_INTERRUPTION_RESUME_DELAY_S = 0.4

_BACKCHANNEL_ONLY_RE = re.compile(
    r"^(hmm+|hm+|uh+|ah+|oh+|ok(ay)?|haan+|ji+|yes|yeah|ach+a+|theek(\s+hai)?"
    r"|bol(o|iye)?|chalo)[\s.,!?]*$",
    re.IGNORECASE,
)
_BACKCHANNEL_ONLY_DEVANAGARI_RE = re.compile(
    r"^(हाँ+|हां+|जी+|हम्म+|ठीक(\s+है)?|अच्छा|ओके|बोलो|बोलिए|चलो)[\s।,!?]*$"
)


def _is_likely_false_interruption(userdata: dict, text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return True
    agent_was_speaking = userdata.get("_agent_state") == "speaking"
    if not agent_was_speaking:
        return False
    return bool(
        _BACKCHANNEL_ONLY_RE.match(stripped)
        or _BACKCHANNEL_ONLY_DEVANAGARI_RE.match(stripped)
    )


_SERVICE_CENTER_PROMPT = build_prompt(f"""
=========================================
AUTHORIZED ŠKODA CAR SERVICE CENTER WORKFLOW
=========================================
You are LILY, a female Service Advisor representing {SERVICE_CENTER_NAME}.
Your ONLY job is to help the customer identify their vehicle and book a service appointment slot.

Follow this EXACT conversation flow for every call:

1. GREETING & IDENTIFY INTENT:
   - Warmly greet the customer: "Namaste! Škoda Service Center mein aapka swagat hai. Main aapki car service appointment book karne mein help kar sakti hoon. Aap kis car ke liye appointment lena chahenge?"
   - Listen to customer's intent.

2. VEHICLE IDENTIFICATION (Škoda-only — this must be fully resolved before Step 3):
   - Ask for the Vehicle Registration Number (e.g., RJ14CA1234) OR Registered Mobile Number.
   - Immediately call lookup_vehicle_by_registration or lookup_customer_by_phone.
   - If Vehicle Found: confirm vehicle details with customer (Brand, Model, e.g., "Škoda Rapid").
   - If registration lookup fails, or the customer has no registration number:
     * Do NOT apologize (e.g. never say "maaf kijiye aapka registered number nahi mil paa raha hai") and do NOT ask the customer to repeat or retry the registration number. Move on immediately.
     * Follow this exact order for an unregistered vehicle: Brand → Model → Year → (then Step 3 customer check) → Date → Preferred Time → check the slot → confirm → book.
     * Call collect_unregistered_vehicle_details as soon as the customer mentions ANY vehicle detail (even just brand or model alone), and again each time they give more.
     * Only Brand, Model, and Year are needed — do NOT ask for the variant.
     * Do NOT proceed to Step 3 or beyond until that tool confirms Brand, Model, and Year are all collected AND the brand is Škoda.
   - If the vehicle's brand is identified as anything other than Škoda, clearly explain that only Škoda vehicles are serviced here, STOP immediately, and do not continue booking for that vehicle — do not ask for further details, do not offer slots/booking for it. Never assume a non-Škoda vehicle is actually a Škoda.

3. CUSTOMER INFORMATION CHECK:
   - Call check_customer_profile to check for missing required fields — only Name and Phone are ever required.
   - If incomplete, ask ONLY for the missing field(s) and save via update_customer_profile.
   - Do NOT ask for email address or home address — they are never required, especially for a new/unregistered customer.
   - If complete, continue directly.

4. SERVICE TYPE — DO NOT ASK. This step does not exist.
   - Never ask "what service are you looking for" or anything similar.
   - Never mention or list Periodic Maintenance, Repair, Electrical & AC, Detailing, or Basic/Standard/Major Service on your own.
   - After vehicle + customer identification, go DIRECTLY to asking for the customer's preferred appointment date and time (Step 5). Treat every booking as a standard periodic-service appointment unless the customer volunteers a specific symptom/complaint (e.g. AC not cooling, engine noise) — only then use get_symptom_recommendation, and only if they explicitly ask what a service includes should you give ONE short answer via get_service_catalog.

5. DATE & TIME — ASK FIRST, DON'T LIST PROACTIVELY, NEVER AUTO-SUGGEST ALTERNATIVES:
   - Always ask the customer for their preferred date AND time first (e.g., "Aapko kis date aur time par aana sahi rahega?"). Do not list available slots yet.
   - If they give a date but no time, ask for their preferred time before checking anything.
   - Once you have a specific date+time, call get_available_slots with that date AND requested_time set to exactly what the customer said. Do NOT read out the full list of slots — the tool returns exactly one of three outcomes, and its wording tells you precisely what to say:
     * AVAILABLE: confirm just that one time with the customer.
     * OUTSIDE WORKING HOURS: tell them clearly this time is outside working hours (state the working hours it returned). Do NOT suggest another time yourself.
     * WITHIN WORKING HOURS BUT NO SLOT: tell them clearly this time is within working hours but that specific slot isn't available. Do NOT suggest another time yourself.
   - NEVER change the customer's requested time automatically, and NEVER suggest another time on your own in either "outside hours" or "no slot" case — simply state the fact and then wait. Only offer alternatives (call get_available_slots again without requested_time to get the full list) if the customer explicitly asks for options ("aur koi time?", "kya options hain") or clearly agrees when you ask if they'd like to hear some.
   - Only read out a full list of open slots if the customer explicitly asks "what slots are available" / "kya options hain" — then call get_available_slots for the relevant date (no requested_time) and share the options.
   - If the customer has no date preference ("koi bhi date", "any date is fine"): first check if today (per current IST time) still has working hours and available slots remaining — call get_available_slots with today's date. If nothing remains today, move to the next available future date and offer its earliest slot.
   - WORKING HOURS vs. AVAILABLE SLOTS — THESE ARE TWO DIFFERENT THINGS, NEVER CONFUSE THEM:
     * "Workshop working hours" (e.g. 09:00 AM - 06:00 PM) is when the workshop is open. "Available appointment slots" are the specific bookable times within those hours — get_available_slots reports both, separately, every time.
     * A time can be INSIDE working hours but still have NO appointment slot available at that exact time. That is a slot-availability gap, NOT the workshop closing early — never say "workshop is only open till X" when X is actually just the last available slot.
     * Before creating a booking, both working hours AND actual slot availability must hold — create_service_booking itself validates both and will refuse (with the specific reason) if either fails, so always trust and relay its actual refusal reason rather than guessing.
     * Never state a different closing time for different dates unless get_available_slots' working_hours value actually differs for that date — always read the exact working_hours it returns rather than assuming or recalling from a previous date's answer.
   - AM/PM PRECISION: be exact — 12:00 AM = midnight (start of day), 12:00 PM = noon (midday), 12:30 AM = 00:30, 12:30 PM = 12:30. Never guess if ambiguous; ask the customer to clarify (e.g. "raat 12 baje" vs "dopeher 12 baje").

6. BOOKING SUMMARY & CONFIRMATION:
   - Call generate_booking_summary with the confirmed service(s), date, and time to read back a complete summary.
   - Read the summary clearly to the customer and ask for explicit confirmation: "Kya main is booking ko confirm kar doon?"
   - Only create the booking after the customer EXPLICITLY confirms the selected date and time — an ambiguous or unclear reply is not confirmation; ask again.

7. CREATE BOOKING & SEND CONFIRMATION:
   - Only upon explicit customer confirmation, call create_service_booking with the SAME service(s)/date/time as the summary.
   - If create_service_booking refuses (e.g. slot unavailable, past time, incomplete vehicle info), read the ACTUAL reason it gave back to the customer in your own words, and guide them to a valid alternative — do NOT just repeat "I can't confirm the booking" or ask a generic "how else can I help" without explaining why.
   - Call send_booking_confirmation to dispatch SMS / WhatsApp / Email confirmation.

8. AFTER BOOKING — STAY HELPFUL, DON'T MISREAD FOLLOW-UPS:
   - Keep helping with anything else the customer asks — address, workshop timings, confirmation details, pricing, etc.
   - A question about address, timing, or booking details is NOT a request to reschedule. Only treat it as a reschedule if the customer explicitly asks to change the date/time of their booking (e.g. "date change kar do", "reschedule karna hai").
   - Remember what you've already answered this call — never re-ask something already covered, and never lose track of a question you just asked (if the customer's reply is unclear, re-ask the SAME question rather than switching to a different one).

9. CLOSING:
   - Follow the shared CALL CLOSURE sequence (call ask_anything_else, then record_call_outcome) before ending the call.

=========================================
VOICE & LATENCY RULES
=========================================
- Keep responses concise and suitable for voice (avoid long paragraphs).
- Ask ONE question at a time.
- Always match the caller's language (English, Hindi, Hinglish) turn-by-turn.
- Never invent service charges, slot times, or working hours without a tool call.
""")

ALL_SERVICE_TOOLS = [
    lookup_vehicle_by_registration,
    lookup_customer_by_phone,
    collect_unregistered_vehicle_details,
    check_customer_profile,
    update_customer_profile,
    get_service_catalog,
    get_symptom_recommendation,
    get_available_slots,
    create_service_booking,
    generate_booking_summary,
    send_booking_confirmation,
    record_customer_intent,
    handle_objection,
    switch_language,
    escalate_to_human,
    ask_anything_else,
    record_call_outcome,
]


async def _apply_language_update(agent: Agent, userdata: dict, base_instructions: str, text: str) -> None:
    try:
        call_state: Optional[CallState] = userdata.get("call_state")
        if call_state is None or not text or not text.strip():
            return

        lang = resolve_current_language(call_state, text)
        if getattr(agent, "_applied_language", None) == lang:
            return

        lock = getattr(agent, "_lang_update_lock", None)
        if lock is None:
            lock = asyncio.Lock()
            agent._lang_update_lock = lock

        async with lock:
            if getattr(agent, "_applied_language", None) == lang:
                return

            agent._applied_language = lang
            await agent.update_instructions(base_instructions + language_instruction_trailer(lang))

    except Exception:
        logger.exception("Language instruction update failed; call continues on last-known instructions.")


def _on_interim_transcript_for_language(userdata: dict, ev) -> None:
    if not getattr(ev, "is_final", False):
        return
    agent = userdata.get("_active_agent")
    text = getattr(ev, "transcript", None)
    if agent is None or not text:
        return
    base_instructions = getattr(agent, "_base_instructions", None)
    if base_instructions is None:
        return
    task = asyncio.create_task(_apply_language_update(agent, userdata, base_instructions, text))
    bg_tasks = getattr(agent, "_bg_tasks", None)
    if bg_tasks is not None:
        bg_tasks.add(task)
        task.add_done_callback(bg_tasks.discard)


def _attach_interim_language_listener(session: AgentSession, userdata: dict) -> None:
    session.on("user_input_transcribed", lambda ev: _on_interim_transcript_for_language(userdata, ev))


def _attach_agent_state_listener(session: AgentSession, userdata: dict) -> None:
    def _on_agent_state_changed(ev) -> None:
        userdata["_agent_state"] = getattr(ev, "new_state", None)

    session.on("agent_state_changed", _on_agent_state_changed)


class _LanguageAwareAgent(Agent):
    def __init__(self, instructions: str, chat_ctx: ChatContext, tools: list, userdata: dict):
        super().__init__(instructions=instructions, chat_ctx=chat_ctx, tools=tools)
        self.userdata = userdata
        self._base_instructions = instructions
        self._bg_tasks: set = set()
        self._applied_language: Optional[str] = None
        self._lang_update_lock: asyncio.Lock = asyncio.Lock()
        self._turn_seq: int = 0

    async def on_user_turn_completed(self, chat_ctx: ChatContext, new_message: ChatMessage) -> None:
        text = getattr(new_message, "text_content", None) or str(getattr(new_message, "content", "") or "")
        self._turn_seq += 1
        call_state: Optional[CallState] = self.userdata.get("call_state")
        if call_state is not None:
            call_state.turn_seq += 1

        if _is_likely_false_interruption(self.userdata, text):
            seq = self._turn_seq
            _r = asyncio.create_task(self._resume_after_false_interruption(seq))
            self._bg_tasks.add(_r)
            _r.add_done_callback(self._bg_tasks.discard)
            raise StopResponse()

        _t = asyncio.create_task(_apply_language_update(self, self.userdata, self._base_instructions, text))
        self._bg_tasks.add(_t)
        _t.add_done_callback(self._bg_tasks.discard)
        await super().on_user_turn_completed(chat_ctx, new_message)

    async def _resume_after_false_interruption(self, seq: int) -> None:
        try:
            await asyncio.sleep(_FALSE_INTERRUPTION_RESUME_DELAY_S)
            if seq != self._turn_seq:
                return
            await self.session.generate_reply(
                instructions=(
                    "That was background noise or a stray sound, not the "
                    "customer speaking or asking something new. "
                    "Naturally continue exactly where you left off in your "
                    "PREVIOUS response — do not restart it or repeat."
                )
            )
        except Exception:
            logger.exception("Resume-after-false-interruption failed.")


class ServiceAdvisorAgent(_LanguageAwareAgent):
    def __init__(self, chat_ctx: ChatContext, userdata: dict, extra_instructions: str = ""):
        instructions = _SERVICE_CENTER_PROMPT
        if extra_instructions:
            instructions = instructions + "\n\n" + extra_instructions
        super().__init__(
            instructions=instructions,
            chat_ctx=chat_ctx,
            tools=ALL_SERVICE_TOOLS,
            userdata=userdata,
        )


def prewarm(proc: agents.JobProcess):
    proc.userdata["db_data"] = load_all_car_service_data()
    logger.info("Worker prewarm completed: all car service center records loaded.")


def extract_caller_phone(participant) -> str:
    if participant is not None:
        attrs = getattr(participant, "attributes", None) or {}
        for key in ("sip.phoneNumber", "sip.from", "sip.callerNumber", "phone_number"):
            value = attrs.get(key)
            if value:
                return value
        identity = getattr(participant, "identity", "") or ""
        if identity:
            return identity
    return os.environ.get("TEST_CALLER_PHONE", "")


def _persist_outcome_and_clear(data: dict, call_state: CallState, user_id: str):
    entry = call_state.call_history_entry
    cust = call_state.customer_record
    if entry and cust:
        customer_id = cust.get("customer_id", "")
        if customer_id:
            persist_booking_call_history(data, customer_id, entry)
    clear_call_state(user_id)


def _build_realtime_session(instructions: str, userdata: dict) -> AgentSession:
    realtime_model = RealtimeModel(
        model="gemini-live-2.5-flash-native-audio",
        voice="Vindemiatrix",
        temperature=0.6,
        instructions=instructions,
        realtime_input_config=_REALTIME_INPUT_CONFIG,
        **_CONTEXT_PRESERVATION_KWARGS,
    )
    session = AgentSession(
        llm=realtime_model,
        userdata=userdata,
        turn_handling={
            "endpointing": {
                "min_delay": 0.5,
            },
            "interruption": {
                "min_duration": 0.3,
            }
        }
    )
    _attach_interim_language_listener(session, userdata)
    _attach_agent_state_listener(session, userdata)
    return session


def _generate_opening_reply_when_ready(session: AgentSession, **kwargs) -> None:
    def _session_running() -> bool:
        for attr in ("_started", "started", "_running", "running"):
            try:
                v = getattr(session, attr)
            except Exception:
                continue
            if isinstance(v, bool):
                return v
        return True

    async def _loop():
        last_exc: Optional[Exception] = None
        for _ in range(80):
            if _session_running():
                try:
                    session.generate_reply(**kwargs)
                    return
                except Exception as e:
                    last_exc = e
            await asyncio.sleep(0.05)
        logger.error("Could not schedule opening reply after 4s: %s", last_exc)

    asyncio.create_task(_loop())


def _register_close_handler(session: AgentSession, on_close) -> None:
    session.on("close", lambda *_: asyncio.create_task(on_close()))


async def entrypoint(ctx: agents.JobContext):
    await ctx.connect(auto_subscribe=agents.AutoSubscribe.AUDIO_ONLY)

    raw_metadata = getattr(ctx.job, "metadata", "") or "{}"
    try:
        job_meta = json.loads(raw_metadata)
    except (json.JSONDecodeError, TypeError):
        job_meta = {}

    is_outbound = job_meta.get("mode") == "outbound"
    is_direct = job_meta.get("mode") == "direct_department"
    db_data = ctx.proc.userdata.get("db_data") if hasattr(ctx, "proc") else None
    if not db_data:
        db_data = load_all_car_service_data()

    if is_outbound:
        await _run_outbound_call(ctx, db_data, job_meta)
    elif is_direct:
        await _run_direct_department_call(ctx, db_data, job_meta)
    else:
        await _run_inbound_call(ctx, db_data)


async def _run_inbound_call(ctx: agents.JobContext, db_data: dict):
    logger.info(f"Connecting INBOUND room {ctx.room.name}")
    caller_participant = await ctx.wait_for_participant()
    user_id = caller_participant.identity if caller_participant and caller_participant.identity else "anonymous"
    caller_phone = extract_caller_phone(caller_participant)

    # Force a FRESH call state, even if a stale one exists for this same
    # identity from a previous call/test that didn't close cleanly (e.g.
    # Playground reconnect) — prevents vehicle/customer data leaking across
    # calls.
    clear_call_state(user_id)
    call_state = get_call_state(user_id)
    call_state.is_outbound = False

    # Match customer by caller phone for PERSONALIZATION ONLY (name in the
    # greeting). Deliberately does NOT set call_state.vehicle_record — the
    # vehicle being serviced must always come from what the customer says,
    # never assumed from their registered vehicle on file.
    cust = find_customer_by_phone(db_data, caller_phone)
    if cust:
        call_state.customer_record = cust

    _userdata = {
        "user_id": user_id,
        "db_data": db_data,
        "call_state": call_state,
        "caller_phone": caller_phone,
        "call_type": "service",
    }

    session = _build_realtime_session(_SERVICE_CENTER_PROMPT, _userdata)
    agent = ServiceAdvisorAgent(chat_ctx=ChatContext(), userdata=_userdata)
    _userdata["_active_agent"] = agent
    await session.start(room=ctx.room, agent=agent, room_input_options=_ROOM_INPUT_OPTIONS)

    customer_name = call_state.customer_record.get("name") if call_state.customer_record else None
    if customer_name:
        greeting = (
            "This is an INBOUND call. Caller's phone matched an existing customer — use their name "
            "naturally for personalization, but do NOT mention or assume any specific vehicle yet.\n\n"
            f"Warmly greet them first in HINGLISH/HINDI as Lily, representing {SERVICE_CENTER_NAME}, "
            f"using their name ({customer_name}). Then ask which vehicle and what service they need "
            "today — do not assume it's their registered vehicle until they confirm."
        )
    else:
        greeting = (
            "This is an INBOUND call for car service booking.\n\n"
            f"Warmly greet them first in HINGLISH/HINDI as Lily, representing {SERVICE_CENTER_NAME}.\n"
            "Say: 'Namaste! Škoda Service Center mein aapka swagat hai. Main aapki car service appointment book karne mein help kar sakti hoon. Aap kis car ke liye appointment lena chahenge?'"
        )

    _generate_opening_reply_when_ready(session, user_input=greeting)

    async def _on_session_close():
        _persist_outcome_and_clear(db_data, call_state, _userdata["user_id"])

    _register_close_handler(session, _on_session_close)


async def _run_outbound_call(ctx: agents.JobContext, db_data: dict, job_meta: dict):
    purpose = job_meta.get("purpose", "service_reminder")
    vehicle_id = job_meta.get("vehicle_id", "")

    logger.info(f"Connecting OUTBOUND room {ctx.room.name} purpose={purpose} vehicle_id={vehicle_id}")

    try:
        caller_participant = await asyncio.wait_for(ctx.wait_for_participant(), timeout=30)
    except asyncio.TimeoutError:
        logger.error("Outbound room timeout — SIP participant did not join.")
        return

    user_id = f"outbound-{vehicle_id}" if vehicle_id else (caller_participant.identity or "outbound-unknown")

    target_vehicle = None
    target_customer = None
    for v in db_data.get("vehicles", []):
        if v.get("vehicle_id") == vehicle_id:
            target_vehicle = v
            target_customer = find_customer_by_id(db_data, v.get("customer_id"))
            break

    # Force a FRESH call state for this vehicle_id-derived identity too.
    clear_call_state(user_id)
    call_state = get_call_state(user_id)
    call_state.is_outbound = True
    call_state.outbound_purpose = purpose
    call_state.vehicle_record = target_vehicle
    call_state.customer_record = target_customer

    _userdata = {
        "user_id": user_id,
        "db_data": db_data,
        "call_state": call_state,
        "caller_phone": target_customer.get("phone", "") if target_customer else "",
        "call_type": "service",
    }

    extra_instructions = (
        "### Outbound Call Context\n"
        "This is an OUTBOUND service reminder call — the service center is proactively calling "
        f"the customer ({target_customer.get('name', 'Customer') if target_customer else 'Customer'}) "
        f"regarding periodic maintenance for their vehicle {target_vehicle.get('brand', '')} {target_vehicle.get('model', '')} "
        f"({target_vehicle.get('registration_number', '') if target_vehicle else ''}). "
        "Warmly greet them, state that their service is due, and offer available time slots."
    )

    session = _build_realtime_session(_SERVICE_CENTER_PROMPT, _userdata)
    agent = ServiceAdvisorAgent(chat_ctx=ChatContext(), userdata=_userdata, extra_instructions=extra_instructions)
    _userdata["_active_agent"] = agent
    await session.start(room=ctx.room, agent=agent, room_input_options=_ROOM_INPUT_OPTIONS)

    opener = (
        f"Begin the OUTBOUND service reminder call now. Greet warmly as Lily from {SERVICE_CENTER_NAME}, "
        f"and state that you're calling regarding periodic maintenance due for their vehicle "
        f"{target_vehicle.get('brand', '')} {target_vehicle.get('model', '')} ({target_vehicle.get('registration_number', '') if target_vehicle else ''})."
    )

    _generate_opening_reply_when_ready(session, user_input=opener)

    async def _on_session_close():
        _persist_outcome_and_clear(db_data, call_state, user_id)
        if vehicle_id:
            mark_service_reminder_sent(db_data, vehicle_id)

    _register_close_handler(session, _on_session_close)


async def _run_direct_department_call(ctx: agents.JobContext, db_data: dict, job_meta: dict):
    caller_participant = await ctx.wait_for_participant()
    user_id = caller_participant.identity if caller_participant and caller_participant.identity else "direct-service"

    clear_call_state(user_id)
    call_state = get_call_state(user_id)
    call_state.is_outbound = False

    _userdata = {
        "user_id": user_id,
        "db_data": db_data,
        "call_state": call_state,
        "caller_phone": "",
        "call_type": "service",
    }

    session = _build_realtime_session(_SERVICE_CENTER_PROMPT, _userdata)
    agent = ServiceAdvisorAgent(chat_ctx=ChatContext(), userdata=_userdata)
    _userdata["_active_agent"] = agent
    await session.start(room=ctx.room, agent=agent, room_input_options=_ROOM_INPUT_OPTIONS)

    _generate_opening_reply_when_ready(
        session,
        user_input=f"Begin the call now as Lily representing {SERVICE_CENTER_NAME}. Greet the caller and ask how you can help."
    )

    async def _on_session_close():
        _persist_outcome_and_clear(db_data, call_state, _userdata["user_id"])

    _register_close_handler(session, _on_session_close)


if __name__ == "__main__":
    agents.cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            agent_name=UNIFIED_AGENT_NAME,
            load_threshold=0.70,
            num_idle_processes=4,
        )
    )