"""
services.py — All car-service-center domain logic and tools in one file.

Combines what was previously split across services/data_service.py,
services/vehicle_service.py, services/service_catalog_service.py, and
services/booking_service.py into a single module, per request. Pickup &
drop has been removed entirely (no pickup_areas data, no pickup fields on
bookings, no pickup step in the booking flow) — see triage_agent.py and
common.py for the matching prompt-side removal.
"""

import json
import logging
import os
from datetime import datetime
from typing import Optional

from livekit.agents import RunContext, function_tool

from common import CallState, now_ist, today_ist_str, IST, cancel_stale_confirmation

logger = logging.getLogger("services")

# ---------------------------------------------------------------------------
# DEMO working-hours rule (single source of truth):
#   Monday through Sunday, 09:00 AM - 06:00 PM inclusive.
#   No per-day exceptions (no Saturday/Sunday/holiday/half-day logic).
#   The ONLY availability restriction is past date/time (IST). There is NO
#   slot-capacity / bay-occupancy / "already booked" checking in this demo —
#   any future in-hours date/time is accepted.
# ---------------------------------------------------------------------------

_WORKSHOP_WORKING_HOURS = "09:00 AM - 06:00 PM"
_WORKSHOP_OPEN_TIME_STR = "09:00 AM"
_WORKSHOP_CLOSE_TIME_STR = "06:00 PM"


def _parse_time_of_day(time_str: str):
    """Parse a bare time-of-day string ('6 PM', '6:00 PM', '06:00 PM') into a time object. Returns None on failure."""
    cleaned = (time_str or "").strip().upper()
    for fmt in ("%I:%M %p", "%I %p", "%I.%M %p"):
        try:
            return datetime.strptime(cleaned, fmt).time()
        except ValueError:
            continue
    return None


def _normalize_time_for_compare(time_str: str) -> str:
    """
    Normalize a time string for equality comparisons so harmless formatting
    differences ('9:00 AM' vs '09:00 AM', extra spaces, case) never cause a
    false anti-hallucination mismatch. Falls back to a simple
    lowercased/stripped string if it can't be parsed as a time at all.
    """
    parsed = _parse_time_of_day(time_str)
    if parsed is None:
        return " ".join((time_str or "").strip().lower().split())
    return parsed.strftime("%I:%M %p")


def _normalize_text_for_compare(text: str) -> str:
    """Normalize free text (service names/ids, etc.) for tolerant comparison — collapses whitespace, lowercases."""
    return " ".join((text or "").strip().lower().split())


def _is_within_working_hours(time_str: str) -> Optional[bool]:
    """
    True if time_str falls within [09:00 AM, 06:00 PM] INCLUSIVE (so exactly
    06:00 PM is accepted), False if outside, None if time_str couldn't be
    parsed at all (caller should ask the customer to clarify rather than
    assume either way). Same rule every day — Monday through Sunday, no
    exceptions.
    """
    t = _parse_time_of_day(time_str)
    if t is None:
        return None
    open_t = _parse_time_of_day(_WORKSHOP_OPEN_TIME_STR)
    close_t = _parse_time_of_day(_WORKSHOP_CLOSE_TIME_STR)
    return open_t <= t <= close_t


# DEMO/REFERENCE DATA ONLY — NOT a whitelist of allowed booking times and NOT
# used for any availability/rejection decision. It exists purely so
# get_available_slots has some example times to show a customer who asks for
# "options" without naming an exact time. Booking validation is based solely
# on _is_within_working_hours() + past-date/time checks, below.
_STANDARD_SLOT_TEMPLATE = [
    ("09:00 AM", "Bay 1"),
    ("10:30 AM", "Bay 2"),
    ("12:00 PM", "Bay 3"),
    ("02:00 PM", "Bay 1"),
    ("03:30 PM", "Bay 2"),
    ("04:00 PM", "Bay 3"),
    ("05:00 PM", "Bay 1"),
]


def _parse_slot_datetime(date_str: str, time_str: str):
    """Parse a 'YYYY-MM-DD' + '09:00 AM'-style time into an IST-aware datetime. Returns None on bad input."""
    try:
        naive = datetime.strptime(f"{(date_str or '').strip()} {(time_str or '').strip()}", "%Y-%m-%d %I:%M %p")
        return naive.replace(tzinfo=IST)
    except ValueError:
        return None


def _is_past_slot(date_str: str, time_str: str) -> bool:
    """
    True if this date+time is strictly in the past relative to the current
    IST time. This is the ONLY availability restriction in the demo besides
    working hours — no slot-capacity checking exists.
    """
    dt = _parse_slot_datetime(date_str, time_str)
    if dt is None:
        return False  # can't parse -> let normal validation elsewhere handle it
    return dt <= now_ist()


def _is_past_date(date_str: str) -> bool:
    """True if this date (ignoring time-of-day) is strictly before today, IST."""
    try:
        target = datetime.strptime((date_str or "").strip(), "%Y-%m-%d").date()
    except ValueError:
        return False
    return target < now_ist().date()

# ---------------------------------------------------------------------------
# Data paths & loading (formerly services/data_service.py)
# ---------------------------------------------------------------------------

CUSTOMER_DATA_PATH = os.environ.get("CUSTOMER_DATA_PATH", "all_json/customer_data.json")
VEHICLE_DATA_PATH = os.environ.get("VEHICLE_DATA_PATH", "all_json/vehicle_data.json")
SERVICE_CATALOG_PATH = os.environ.get("SERVICE_CATALOG_PATH", "all_json/service_catalog.json")
SLOTS_PATH = os.environ.get("SLOTS_PATH", "all_json/slots.json")
BOOKINGS_PATH = os.environ.get("BOOKINGS_PATH", "all_json/bookings.json")


def _load_json(path: str, default=None):
    if default is None:
        default = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error("Data file not found: %s", path)
        return default
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON in %s: %s", path, exc)
        return default


def load_all_car_service_data() -> dict:
    """Load all JSON databases into memory (pickup areas excluded)."""
    customer_db = _load_json(CUSTOMER_DATA_PATH, {"customers": []})
    vehicle_db = _load_json(VEHICLE_DATA_PATH, {"vehicles": [], "unregistered_vehicle_template": {}})
    catalog_db = _load_json(SERVICE_CATALOG_PATH, {"categories": []})
    slots_db = _load_json(SLOTS_PATH, {"workshop_calendar": [], "bay_info": []})
    bookings_db = _load_json(BOOKINGS_PATH, {"bookings": []})

    customers = customer_db.get("customers", [])
    vehicles = vehicle_db.get("vehicles", [])

    vehicles_by_id = {v["vehicle_id"]: v for v in vehicles if v.get("vehicle_id")}
    vehicles_by_reg = {
        v["registration_number"].upper().replace(" ", ""): v
        for v in vehicles
        if v.get("registration_number")
    }
    customers_by_id = {c["customer_id"]: c for c in customers if c.get("customer_id")}

    return {
        "customers": customers,
        "vehicles": vehicles,
        "vehicles_by_id": vehicles_by_id,
        "vehicles_by_reg": vehicles_by_reg,
        "customers_by_id": customers_by_id,
        "unregistered_vehicle_template": vehicle_db.get("unregistered_vehicle_template", {}),
        "service_catalog": catalog_db.get("categories", []),
        "workshop_calendar": slots_db.get("workshop_calendar", []),
        "bay_info": slots_db.get("bay_info", []),
        "bookings": bookings_db.get("bookings", []),
    }


def find_vehicle_by_registration(data: dict, reg_number: str) -> Optional[dict]:
    """Find a vehicle record by registration number (case-insensitive, space-ignorant)."""
    if not reg_number:
        return None
    normalized = reg_number.upper().replace(" ", "").replace("-", "")
    for v in data.get("vehicles", []):
        v_reg = (v.get("registration_number") or "").upper().replace(" ", "").replace("-", "")
        if v_reg == normalized:
            return v
    return None


def find_customer_by_phone(data: dict, phone: str) -> Optional[dict]:
    """Find customer by last 10 digits of phone number."""
    if not phone:
        return None
    normalized = "".join(ch for ch in phone if ch.isdigit())
    if not normalized:
        return None
    target = normalized[-10:] if len(normalized) >= 10 else normalized
    for c in data.get("customers", []):
        c_phone = "".join(ch for ch in c.get("phone", "") if ch.isdigit())
        if c_phone and c_phone.endswith(target):
            return c
    return None


def find_customer_by_id(data: dict, customer_id: str) -> Optional[dict]:
    """Find customer record by customer_id."""
    if not customer_id:
        return None
    cid = customer_id.strip().lower()
    for c in data.get("customers", []):
        if (c.get("customer_id") or "").strip().lower() == cid:
            return c
    return None


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def persist_customer_data(data: dict, path: str = CUSTOMER_DATA_PATH) -> bool:
    try:
        payload = {"customers": data.get("customers", [])}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.info("Persisted customer data to %s", path)
        return True
    except OSError as exc:
        logger.error("Failed to write customer data: %s", exc)
        return False


def persist_vehicle_data(data: dict, path: str = VEHICLE_DATA_PATH) -> bool:
    try:
        payload = {
            "vehicles": data.get("vehicles", []),
            "unregistered_vehicle_template": data.get("unregistered_vehicle_template", {}),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.info("Persisted vehicle data to %s", path)
        return True
    except OSError as exc:
        logger.error("Failed to write vehicle data: %s", exc)
        return False


def persist_bookings(data: dict, path: str = BOOKINGS_PATH) -> bool:
    try:
        payload = {"bookings": data.get("bookings", [])}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.info("Persisted bookings data to %s", path)
        return True
    except OSError as exc:
        logger.error("Failed to write bookings data: %s", exc)
        return False


def persist_slots(data: dict, path: str = SLOTS_PATH) -> bool:
    try:
        payload = {
            "workshop_calendar": data.get("workshop_calendar", []),
            "bay_info": data.get("bay_info", []),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.info("Persisted slots data to %s", path)
        return True
    except OSError as exc:
        logger.error("Failed to write slots data: %s", exc)
        return False


def persist_booking_call_history(data: dict, customer_id: str, entry: dict) -> bool:
    customer = find_customer_by_id(data, customer_id)
    if not customer:
        return False
    customer.setdefault("call_history", []).append(entry)
    return persist_customer_data(data)


def mark_service_reminder_sent(data: dict, vehicle_id: str) -> bool:
    for v in data.get("vehicles", []):
        if v.get("vehicle_id") == vehicle_id:
            v["reminder_sent_date"] = today_ist_str()
            return persist_vehicle_data(data)
    return False


def list_vehicles_needing_service_reminder(data: dict) -> list[dict]:
    """Vehicles due for service (last service >= ~180 days ago) not yet reminded today."""
    today = now_ist().date()
    results = []
    for v in data.get("vehicles", []):
        last_date_str = v.get("last_service_date", "")
        if not last_date_str:
            continue
        try:
            last_date = datetime.strptime(last_date_str, "%Y-%m-%d").date()
        except ValueError:
            continue
        if (today - last_date).days >= 180:
            if v.get("reminder_sent_date") != today_ist_str():
                results.append(v)
    return results


# ---------------------------------------------------------------------------
# Vehicle & customer tools (formerly services/vehicle_service.py)
# ---------------------------------------------------------------------------

@function_tool
async def lookup_vehicle_by_registration(ctx: RunContext, registration_number: str) -> str:
    """
    Look up a vehicle in the dealer database using its registration number (e.g. RJ14CA1234).
    If found, loads both the vehicle record and its linked customer profile into session state.
    """
    data = ctx.userdata["db_data"]
    state: CallState = ctx.userdata["call_state"]
    cancel_stale_confirmation(state)

    rec = find_vehicle_by_registration(data, registration_number)
    if not rec:
        state.log(f"No vehicle found for registration: {registration_number}")
        return (
            f"Registration '{registration_number}' is not in our records. Do NOT apologize for this "
            "or dwell on it, and do NOT ask the customer to repeat/retry the registration number. "
            "Simply move on to identifying the vehicle directly: call collect_unregistered_vehicle_details "
            "now with whatever they've already told you, and continue asking for Brand, Model, and Year "
            "as needed per the unregistered-vehicle flow."
        )

    if (rec.get("brand") or "").strip().lower() != "skoda":
        state.vehicle_record = None
        state.vehicle_brand_blocked = True
        state.blocked_vehicle_label = f"{rec.get('brand', '')} {rec.get('model', '')}".strip()
        state.log(f"BLOCKED: registration {registration_number} is a non-Skoda vehicle ({state.blocked_vehicle_label}).")
        return (
            f"This registration belongs to a {state.blocked_vehicle_label}, which is NOT a Skoda. "
            "This is a Skoda-only service center. Politely decline in the caller's language, do not "
            "reveal any further details about this vehicle, and do not proceed with service, slots, "
            "or booking for it."
        )

    state.vehicle_record = rec
    state.vehicle_brand_blocked = False
    state.blocked_vehicle_label = None
    customer = find_customer_by_id(data, rec.get("customer_id", ""))
    if not customer and ctx.userdata.get("caller_phone"):
        customer = find_customer_by_phone(data, ctx.userdata["caller_phone"])

    if customer:
        state.customer_record = customer

    state.log(f"Loaded vehicle record {rec.get('vehicle_id')} ({rec.get('brand')} {rec.get('model')}).")

    cust_name = customer.get("name", "Customer") if customer else "Customer"
    return (
        f"Vehicle found and confirmed:\n"
        f"- Registration: {rec.get('registration_number')}\n"
        f"- Owner: {cust_name}\n"
        f"- Brand & Model: {rec.get('brand')} {rec.get('model')} {rec.get('variant')} ({rec.get('year')})\n"
        f"- Fuel Type: {rec.get('fuel_type')}, Color: {rec.get('color')}\n"
        f"- Odometer: {rec.get('odometer_km')} km\n"
        f"- Last Service: {rec.get('last_service_date')} ({rec.get('last_service_type')})\n"
        f"- Warranty Status: {rec.get('warranty_status')}\n"
        f"- AMC Active: {'Yes' if rec.get('amc_active') else 'No'}\n"
        "Confirm this vehicle with the customer and ask for their service requirement."
    )


@function_tool
async def lookup_customer_by_phone(ctx: RunContext, phone: str = "") -> str:
    """
    Look up customer profile in dealer database by registered mobile number.
    If phone is omitted, uses the caller's phone number from the session context.
    """
    data = ctx.userdata["db_data"]
    state: CallState = ctx.userdata["call_state"]
    cancel_stale_confirmation(state)

    search_phone = phone or ctx.userdata.get("caller_phone", "")
    customer = find_customer_by_phone(data, search_phone)

    if not customer:
        state.log(f"No customer found for phone: {search_phone}")
        return (
            f"No customer record found for phone '{search_phone}'. Do NOT apologize for this or "
            "dwell on it. Simply proceed: ask for the Vehicle Registration Number, or if they don't "
            "have one, move directly into collect_unregistered_vehicle_details (Brand, Model, Year). "
            "Only Name and Phone are needed for the customer profile — Email/Address are not required."
        )

    state.customer_record = customer
    state.log(f"Loaded customer profile for {customer.get('name')}.")

    # NOTE: intentionally NOT setting state.vehicle_record here. A matched
    # phone number tells us WHO the customer is, not WHICH vehicle they're
    # calling about today — that must come from what the customer actually
    # says. This list is for personalization/reference only.
    reg_vehicles = customer.get("registered_vehicles", [])
    known_vehicles = [v for v in data.get("vehicles", []) if v.get("vehicle_id") in reg_vehicles]

    response = (
        f"Customer profile found:\n"
        f"- Name: {customer.get('name')}\n"
        f"- Phone: {customer.get('phone')}\n"
        f"- Email: {customer.get('email') or 'Missing'}\n"
        f"- Address: {customer.get('address') or 'Missing'}\n"
        f"- Customer Since: {customer.get('customer_since') or 'N/A'}\n"
        f"- Loyalty Tier: {customer.get('loyalty_tier', 'Standard')}\n"
    )
    if known_vehicles:
        listed = "; ".join(
            f"{v.get('registration_number')} ({v.get('brand')} {v.get('model')} {v.get('variant')})"
            for v in known_vehicles
        )
        response += (
            f"\nRegistered vehicle(s) on file (for reference only — do NOT assume this is the "
            f"vehicle being serviced today): {listed}"
        )
    response += (
        "\nGreet the customer by name for personalization, then ask which vehicle they're calling "
        "about — do not assume it's one of the vehicles listed above until they say so. Do NOT ask "
        "what service they need."
    )
    return response


@function_tool
async def collect_unregistered_vehicle_details(
    ctx: RunContext,
    brand: str = "",
    model: str = "",
    year: str = "",
) -> str:
    """
    Call this whenever the customer has no registration number (or lookup
    failed) and is describing their vehicle by Brand/Model/Year. Call it as
    soon as ANY of these is mentioned — even just the brand or model alone
    — passing only the field(s) just given; call it again as more fields
    are collected. This is a Skoda-only center: as soon as a brand is given
    and it is NOT Skoda, this immediately blocks the vehicle and the flow
    must stop. Only once Brand, Model, and Year are ALL known, and the
    brand is Skoda, does this create the vehicle record that
    service/slots/booking tools require. Do NOT ask for the vehicle
    variant — it is not needed.
    """
    state: CallState = ctx.userdata["call_state"]
    cancel_stale_confirmation(state)
    details = state.pending_vehicle_details

    if brand:
        details["brand"] = brand.strip()
    if model:
        details["model"] = model.strip()
    if year:
        details["year"] = year.strip()

    if details.get("brand") and details["brand"].strip().lower() != "skoda":
        state.vehicle_record = None
        state.vehicle_brand_blocked = True
        state.blocked_vehicle_label = f"{details.get('brand', '')} {details.get('model', '')}".strip()
        state.log(f"BLOCKED: customer's unregistered vehicle is non-Skoda ({state.blocked_vehicle_label}).")
        return (
            f"The customer's vehicle ({state.blocked_vehicle_label}) is NOT a Skoda. This is a "
            "Skoda-only service center. Do NOT ask for further vehicle details, do NOT proceed to "
            "service requirement, slots, or booking. Politely decline in the caller's language and "
            "stop this vehicle's flow."
        )

    required = ["brand", "model", "year"]
    missing = [f for f in required if not details.get(f)]

    if missing:
        state.log(f"Collecting unregistered vehicle details, still missing: {missing}")
        return (
            f"Confirmed Skoda so far. Still missing: {', '.join(missing)}. Ask for ONLY these "
            "remaining fields, one at a time — do NOT ask for variant, it is not needed. Do NOT "
            "proceed to service requirement, slots, or booking until all of Brand, Model, and Year "
            "are collected."
        )

    state.vehicle_record = {
        "vehicle_id": None,
        "registration_number": "UNREGISTERED",
        "brand": details["brand"],
        "model": details["model"],
        "variant": "",
        "year": details["year"],
    }
    state.vehicle_brand_blocked = False
    state.blocked_vehicle_label = None
    state.pending_vehicle_details = {}
    state.log(f"Unregistered Skoda vehicle fully identified: {state.vehicle_record}")
    return (
        f"Vehicle identification complete: Skoda {details['model']} ({details['year']}), no "
        "registration number on file. You may now proceed to the service requirement and booking "
        "steps."
    )


@function_tool
async def check_customer_profile(ctx: RunContext) -> str:
    """
    Check the loaded customer profile for the only two required fields:
    Name and Phone. Email and Address are never required — do not ask for
    them, especially for a new/unregistered customer.
    """
    state: CallState = ctx.userdata["call_state"]
    cancel_stale_confirmation(state)
    cust = state.customer_record

    if not cust:
        return (
            "No customer profile is currently loaded. "
            "Please collect only: Name and Phone from the customer."
        )

    missing = []
    if not cust.get("name"):
        missing.append("Name")
    if not cust.get("phone"):
        missing.append("Phone")

    if missing:
        return f"Customer profile is incomplete. Please ask ONLY for the missing fields: {', '.join(missing)}. Do NOT ask for Email or Address."
    return f"Customer profile is complete (Name: {cust.get('name')}, Phone: {cust.get('phone')})."


@function_tool
async def update_customer_profile(
    ctx: RunContext,
    name: str = "",
    phone: str = "",
    email: str = "",
    address: str = "",
) -> str:
    """
    Save customer information given by the customer. Only Name and Phone
    are ever required/asked for — Email and Address are optional and
    should only be saved here if the customer volunteers them unprompted;
    never ask for them.
    """
    data = ctx.userdata["db_data"]
    state: CallState = ctx.userdata["call_state"]
    cancel_stale_confirmation(state)

    cust = state.customer_record
    if not cust:
        cust_id = f"CUST{len(data.get('customers', [])) + 2001}"
        cust = {
            "customer_id": cust_id,
            "name": name,
            "phone": phone or ctx.userdata.get("caller_phone", ""),
            "email": email,
            "address": address,
            "registered_vehicles": [],
            "preferred_language": state.current_language or "Hinglish",
            "customer_since": now_ist().strftime("%Y-%m-%d"),
            "loyalty_tier": "New",
            "call_history": [],
        }
        data.setdefault("customers", []).append(cust)
        state.customer_record = cust
    else:
        if name:
            cust["name"] = name
        if phone:
            cust["phone"] = phone
        if email:
            cust["email"] = email
        if address:
            cust["address"] = address

    persist_customer_data(data)
    state.log(f"Updated customer profile for {cust.get('name')}.")
    return f"Customer profile updated successfully: Name: {cust.get('name')}, Phone: {cust.get('phone')}, Email: {cust.get('email') or 'N/A'}, Address: {cust.get('address') or 'N/A'}."


# ---------------------------------------------------------------------------
# Service catalog tools (formerly services/service_catalog_service.py)
# ---------------------------------------------------------------------------

@function_tool
async def get_service_catalog(ctx: RunContext, category: str = "") -> str:
    """
    Retrieve available car service packages and repair services.
    Optionally filter by category: 'Periodic Maintenance', 'Repair', 'Electrical & AC', or 'Detailing'.
    """
    data = ctx.userdata["db_data"]
    state: CallState = ctx.userdata["call_state"]
    cancel_stale_confirmation(state)
    categories = data.get("service_catalog", [])

    if category:
        cat_lower = category.strip().lower()
        matched = [c for c in categories if c.get("category", "").lower() == cat_lower]
        if not matched:
            valid_names = [c.get("category") for c in categories]
            return f"Category '{category}' not found. Available categories: {', '.join(valid_names)}"
        categories = matched

    res_lines = ["Authorized Skoda Service Catalog:"]
    for cat in categories:
        res_lines.append(f"\n--- {cat.get('category')} ---")
        for svc in cat.get("services", []):
            line = f"- {svc.get('name')} (ID: {svc.get('service_id')}): Rs.{svc.get('price_range_inr')} | Est. time: {svc.get('duration_minutes')} mins"
            if svc.get("description"):
                line += f"\n  Description: {svc.get('description')}"
            res_lines.append(line)

    return "\n".join(res_lines)


@function_tool
async def get_symptom_recommendation(ctx: RunContext, symptom_description: str) -> str:
    """
    Recommend appropriate service category/package based on customer-described symptoms or issues
    (e.g., 'AC not cooling', 'braking noise', 'car not starting', 'dent on door').
    """
    data = ctx.userdata["db_data"]
    state: CallState = ctx.userdata["call_state"]
    cancel_stale_confirmation(state)
    categories = data.get("service_catalog", [])
    desc_lower = symptom_description.lower().strip()

    matches = []
    for cat in categories:
        for svc in cat.get("services", []):
            symptoms = svc.get("symptoms", [])
            for sym in symptoms:
                if sym.lower() in desc_lower or any(word in desc_lower for word in sym.lower().split()):
                    matches.append((svc, cat.get("category")))
                    break

    if not matches:
        return (
            f"Based on symptom '{symptom_description}', we recommend an Engine Diagnostics / General Checkup (SVC204) "
            "or a Standard Service. Service Advisor can inspect the vehicle."
        )

    res_lines = [f"Recommended services for '{symptom_description}':"]
    seen_ids = set()
    for svc, cat_name in matches:
        sid = svc.get("service_id")
        if sid in seen_ids:
            continue
        seen_ids.add(sid)
        res_lines.append(
            f"- {svc.get('name')} ({cat_name}): Rs.{svc.get('price_range_inr')} - {svc.get('description')}"
        )
    return "\n".join(res_lines)


# ---------------------------------------------------------------------------
# Booking tools (formerly services/booking_service.py) — pickup & drop
# fields/logic removed throughout.
# ---------------------------------------------------------------------------

@function_tool
async def get_available_slots(ctx: RunContext, date: str = "", requested_time: str = "") -> str:
    """
    Check whether a requested date/time is bookable under the DEMO rule:
    Monday-Sunday, 09:00 AM-06:00 PM inclusive, and not in the past (IST).
    There is NO slot-capacity/bay-occupancy checking in this demo — any
    future in-hours date/time is valid.
    Args:
        date: Preferred date in YYYY-MM-DD format.
        requested_time: If the customer named ONE specific time (e.g. '6 PM', '9:30 AM'),
            pass it here to get back a precise classification: PAST, OUTSIDE
            WORKING HOURS, or VALID FUTURE BOOKING. Leave blank to just see
            example times for the day (illustrative only, not exhaustive).
    """
    state: CallState = ctx.userdata["call_state"]
    cancel_stale_confirmation(state)
    if state.vehicle_brand_blocked:
        return (
            f"BLOCKED: {state.blocked_vehicle_label or 'this vehicle'} is not a Skoda. "
            "Do not offer or discuss slots for it."
        )
    if not state.vehicle_record:
        return (
            "The vehicle has not been identified yet this call. Do NOT offer or discuss "
            "date/time yet — first call lookup_vehicle_by_registration or "
            "collect_unregistered_vehicle_details to identify the vehicle (Steps 1-2 of the "
            "flow), then come back to date/time."
        )

    if not date.strip():
        return (
            f"Workshop working hours: {_WORKSHOP_WORKING_HOURS}, Monday through Sunday — no "
            "special days. Ask the customer for a specific date to check, or a specific "
            "date+time to validate directly."
        )

    date_clean = date.strip()
    if _is_past_date(date_clean):
        return (
            f"'{date_clean}' has already passed (today is {today_ist_str()} IST). "
            "Ask the customer for a future date instead — never book or offer a past date."
        )

    if requested_time.strip():
        req_norm = requested_time.strip()
        within_hours = _is_within_working_hours(req_norm)

        if within_hours is None:
            return (
                f"Could not parse the requested time '{requested_time}'. Ask the customer to state "
                "it clearly (e.g. '6 PM', '9:30 AM') before checking again."
            )
        if not within_hours:
            return (
                f"OUTSIDE WORKING HOURS: '{req_norm}' on {date_clean} is outside the workshop's "
                f"working hours ({_WORKSHOP_WORKING_HOURS}). Tell the customer clearly that this "
                "time is outside working hours, and do NOT book it. Do NOT proactively suggest "
                "another time — wait for the customer to either give a new time themselves or "
                "explicitly ask for options."
            )
        if _is_past_slot(date_clean, req_norm):
            return (
                f"PAST: '{req_norm}' on {date_clean} has already passed (today is "
                f"{today_ist_str()} IST). Tell the customer this time has already passed and ask "
                "for a future date/time. Do NOT proactively suggest a specific alternative — wait "
                "for the customer to give a new time or ask for options."
            )
        return (
            f"VALID FUTURE BOOKING: '{req_norm}' on {date_clean} is within working hours "
            f"({_WORKSHOP_WORKING_HOURS}) and is in the future. Confirm just this one time with "
            "the customer — proceed to generate_booking_summary."
        )

    # No specific time given — customer wants to browse. There is no fixed
    # slot list in this demo; show a few illustrative example times only.
    # _STANDARD_SLOT_TEMPLATE is DEMO/REFERENCE DATA ONLY — it is NOT the
    # full set of bookable times. Any time within working hours works.
    example_times = [
        t for t, _ in _STANDARD_SLOT_TEMPLATE
        if not _is_past_slot(date_clean, t)
    ]
    lines = [
        f"Workshop working hours for {date_clean}: {_WORKSHOP_WORKING_HOURS} (Monday-Sunday, no "
        "special days).",
    ]
    if example_times:
        lines.append(
            "Example times (illustrative only — ANY time within working hours is bookable, the "
            f"customer is not limited to these): {', '.join(example_times)}."
        )
    else:
        lines.append(
            "All the illustrative example times for today have already passed, but any future "
            f"time up to {_WORKSHOP_CLOSE_TIME_STR} today is still bookable."
        )
    lines.append(
        "Ask the customer for their exact preferred time and call get_available_slots again with "
        "requested_time set to validate it — do not restrict them to the example times above."
    )
    return "\n".join(lines)


@function_tool
async def generate_booking_summary(
    ctx: RunContext,
    service_names_or_ids: str,
    date: str,
    time_slot: str,
) -> str:
    """
    Build the booking summary from what's been gathered so far (service(s),
    date, time) and read it back to the customer for explicit confirmation
    BEFORE calling create_service_booking. Call this with the actual values
    the customer has chosen in this call — do not call it empty.
    """
    state: CallState = ctx.userdata["call_state"]
    cancel_stale_confirmation(state)
    if state.vehicle_brand_blocked:
        return (
            f"BLOCKED: {state.blocked_vehicle_label or 'this vehicle'} is not a Skoda. "
            "Do not generate a booking summary for it."
        )
    cust = state.customer_record
    veh = state.vehicle_record

    if not cust and not veh:
        return "Booking summary incomplete: Missing customer and vehicle details."
    if not veh:
        return (
            "Booking summary incomplete: vehicle is not yet fully identified. If unregistered, keep "
            "calling collect_unregistered_vehicle_details until Brand, Model, and Year are all known "
            "before generating a summary."
        )

    c_name = cust.get("name", "N/A") if cust else "N/A"
    c_phone = cust.get("phone", "N/A") if cust else "N/A"
    v_reg = veh.get("registration_number", "N/A") if veh else "N/A"
    v_model = f"{veh.get('brand', '')} {veh.get('model', '')} {veh.get('variant', '')}".strip() if veh else "N/A"

    # Remember what was just confirmed so create_service_booking can be
    # sanity-checked against it, and so a repeat summary stays consistent.
    state.booking_draft = {
        "service_names_or_ids": service_names_or_ids,
        "date": date,
        "time": time_slot,
    }

    return (
        f"=== BOOKING SUMMARY ===\n"
        f"- Customer: {c_name} | Phone: {c_phone}\n"
        f"- Vehicle: {v_model} ({v_reg})\n"
        f"- Service Required: {service_names_or_ids}\n"
        f"- Date & Time: {date} at {time_slot}\n"
        "Ask customer: 'Kya main is booking ko confirm kar doon?'"
    )


@function_tool
async def create_service_booking(
    ctx: RunContext,
    date: str,
    time_slot: str,
    service_names_or_ids: str,
) -> str:
    """
    Create a new service booking in the database after the customer has
    explicitly confirmed the summary from generate_booking_summary.
    Args:
        date: Booking date (YYYY-MM-DD)
        time_slot: Time slot string (e.g., '09:00 AM', '10:30 AM') or slot_id
        service_names_or_ids: Selected service name(s) or ID(s)
    """
    data = ctx.userdata["db_data"]
    state: CallState = ctx.userdata["call_state"]
    cancel_stale_confirmation(state)

    cust = state.customer_record
    veh = state.vehicle_record

    if state.vehicle_brand_blocked:
        return (
            f"HARD BLOCK: {state.blocked_vehicle_label or 'this vehicle'} is not a Skoda. "
            "A booking cannot be created for it under any circumstances."
        )
    if not cust:
        return "Cannot create booking: Customer profile is missing. Please collect customer details first."
    if not veh:
        return (
            "HARD BLOCK: Vehicle is not fully identified yet — no registered vehicle was loaded and "
            "unregistered vehicle details (Brand, Model, Year) are not all collected. Call "
            "lookup_vehicle_by_registration or collect_unregistered_vehicle_details until the vehicle "
            "is fully identified before attempting to book."
        )
    if (veh.get("brand") or "").strip().lower() != "skoda":
        return (
            f"HARD BLOCK: loaded vehicle is a {veh.get('brand')}, not a Skoda. "
            "A booking cannot be created for it."
        )
    if not veh.get("vehicle_id") and not all(veh.get(f) for f in ("brand", "model", "year")):
        return (
            "HARD BLOCK: this unregistered vehicle is still missing Brand/Model/Year details. "
            "Call collect_unregistered_vehicle_details until all three are collected before booking."
        )

    # FIX (hallucinated / phantom confirmation): a booking may only be
    # created after generate_booking_summary was actually called THIS call
    # and the customer explicitly confirmed it — never on the strength of
    # the model's own text alone. state.booking_draft is set exclusively by
    # generate_booking_summary, so its absence, or a mismatch against what's
    # being booked now, means no real, read-back summary happened yet.
    # Comparison is NORMALIZED (harmless formatting differences like
    # '9:00 AM' vs '09:00 AM', extra whitespace, or capitalization do NOT
    # count as a mismatch) — but a genuinely different date/time/service
    # still correctly invalidates the old confirmation.
    draft = state.booking_draft
    if not draft or "service_names_or_ids" not in draft:
        return (
            "REFUSED: No booking summary has been generated and read back to the customer yet this "
            "call. Call generate_booking_summary with the exact service(s)/date/time first, read the "
            "summary aloud, get the customer's explicit confirmation, and only then call "
            "create_service_booking with those SAME values."
        )
    if (
        draft.get("date", "").strip() != date.strip()
        or _normalize_time_for_compare(draft.get("time", "")) != _normalize_time_for_compare(time_slot)
        or _normalize_text_for_compare(draft.get("service_names_or_ids", "")) != _normalize_text_for_compare(service_names_or_ids)
    ):
        return (
            "REFUSED: these date/time/service values do not match the summary last read back to the "
            "customer (generate_booking_summary). If anything changed, call generate_booking_summary "
            "again with the NEW values, get explicit confirmation again, then retry."
        )

    # FIX (IST-based past-slot guard): never create a booking for a
    # date/time that has already passed. This is the ONLY availability
    # restriction in this demo besides working hours — preserved as-is.
    if _is_past_date(date.strip()):
        return f"REFUSED: {date.strip()} has already passed (today is {today_ist_str()} IST). Ask for a future date."
    if _is_past_slot(date.strip(), time_slot.strip()):
        return f"REFUSED: {time_slot.strip()} on {date.strip()} has already passed (IST). Ask the customer to pick a future time."

    # DEMO RULE: working hours are Monday-Sunday, 09:00 AM-06:00 PM
    # inclusive, with NO per-day exceptions. This is the only other
    # restriction — there is NO slot-capacity/bay-occupancy/"already
    # booked" check in this demo. Any future in-hours date/time is valid.
    within_hours = _is_within_working_hours(time_slot.strip())
    if within_hours is None:
        return f"REFUSED: '{time_slot}' could not be parsed as a time. Ask the customer to state it clearly (e.g. '6 PM', '9:30 AM')."
    if within_hours is False:
        return (
            f"REFUSED (OUTSIDE WORKING HOURS): '{time_slot.strip()}' on {date.strip()} is outside the "
            f"workshop's working hours ({_WORKSHOP_WORKING_HOURS}). Tell the customer this time is "
            "outside working hours and cannot be booked."
        )

    booking_id = f"BOOK7{len(data.get('bookings', [])) + 500:03d}"
    slot_id = f"SLOT-{date.strip().replace('-', '')}-{time_slot.strip().replace(' ', '').replace(':', '')}"

    booking_entry = {
        "booking_id": booking_id,
        "customer_id": cust.get("customer_id"),
        "vehicle_id": veh.get("vehicle_id"),
        "service_ids": [s.strip() for s in service_names_or_ids.split(",")],
        "date": date.strip(),
        "time": time_slot.strip(),
        "slot_id": slot_id,
        "status": "confirmed",
        "estimated_cost_inr": "As per service catalog",
        "created_at": now_ist().isoformat(),
        "confirmation_sent": {"sms": True, "whatsapp": True, "email": bool(cust.get("email"))},
    }

    data.setdefault("bookings", []).append(booking_entry)
    persist_bookings(data)

    state.booking_draft = booking_entry
    state.log(f"Booking {booking_id} created for {veh.get('registration_number')} on {date} at {time_slot}.")

    return (
        f"Booking successfully created!\n"
        f"- Booking ID: {booking_id}\n"
        f"- Customer: {cust.get('name')} ({cust.get('phone')})\n"
        f"- Vehicle: {veh.get('brand')} {veh.get('model')} ({veh.get('registration_number')})\n"
        f"- Service: {service_names_or_ids}\n"
        f"- Date & Time: {date} at {time_slot}\n"
        f"- Confirmation: Sent via SMS & WhatsApp."
    )


@function_tool
async def send_booking_confirmation(
    ctx: RunContext,
    channels: str = "SMS, WhatsApp",
) -> str:
    """
    Send booking confirmation notification to customer via SMS / WhatsApp / Email.
    """
    state: CallState = ctx.userdata["call_state"]
    cancel_stale_confirmation(state)
    cust = state.customer_record

    # FIX (critical NameError — `phone` was referenced below but never
    # assigned anywhere in this function, so this call crashed on every
    # single successful booking). `phone` MUST come from state.customer_record
    # — the customer loaded for THIS call — never from anywhere else, so a
    # confirmation is never sent to the wrong person.
    if not cust:
        state.log("send_booking_confirmation called with no customer loaded — cannot send.")
        return (
            "Cannot send confirmation: no customer profile is loaded for this call. This should "
            "not happen after a successful create_service_booking — treat this as an error and let "
            "the customer know their booking is saved but confirmation could not be sent right now."
        )

    phone = cust.get("phone", "")
    email = cust.get("email") if cust else None

    channels_sent = [c.strip() for c in channels.split(",")]
    if email:
        channels_sent.append("Email")

    state.log(f"Booking confirmation sent via {', '.join(channels_sent)} to {phone} (customer_id={cust.get('customer_id')}).")

    return (
        f"Booking confirmation message sent successfully to {phone} via {', '.join(channels_sent)}. "
        "Thank the customer for choosing Skoda Service Center!"
    )