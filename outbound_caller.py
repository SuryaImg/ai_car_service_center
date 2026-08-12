"""
outbound_caller.py — Decides WHO to call for car service reminders, then dispatches the
LiveKit worker (triage_agent.py, agent_name=common.UNIFIED_AGENT_NAME) into a fresh room
with metadata that tells it this is an outbound service reminder call, and dials the customer's phone.

Metadata contract:
    {"mode": "outbound", "purpose": "service_reminder", "vehicle_id": "...", "call_type": "service"}

Usage:
    python outbound_caller.py                          # scan + call all eligible records once
    python outbound_caller.py --dry-run                 # show who WOULD be called, dial nothing
"""

from __future__ import annotations

import utils.clients
import utils.http_session
import utils.transport
import utils.session
import argparse
import asyncio
import json
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from dotenv import load_dotenv
from livekit import api

from common import UNIFIED_AGENT_NAME
from services.data_service import load_all_car_service_data, find_customer_by_id
from services.booking_service import list_vehicles_needing_service_reminder

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("outbound-caller")

# ── Config ────────────────────────────────────────────────────────────────────

LIVEKIT_URL = os.environ.get("LIVEKIT_URL", "")
LIVEKIT_API_KEY = os.environ.get("LIVEKIT_API_KEY", "")
LIVEKIT_API_SECRET = os.environ.get("LIVEKIT_API_SECRET", "")
TRUNK_ID = os.environ.get("LIVEKIT_OUTBOUND_TRUNK_ID", "")

AGENT_NAME = os.getenv("AGENT_NAME", UNIFIED_AGENT_NAME)
CALL_CONCURRENCY = int(os.getenv("CALL_CONCURRENCY", "1"))
INDIA_CC = os.getenv("INDIA_COUNTRY_CODE", "+91")
CALL_LOG_FILE = Path(os.getenv("CALL_LOG_FILE", "call_log.jsonl"))

ROOM_PREFIX = "service-outbound-"


def normalise_phone(raw: str) -> Optional[str]:
    raw = str(raw or "").strip()
    digits = "".join(ch for ch in raw if ch.isdigit())
    if not digits:
        return None

    if len(digits) == 11 and digits.startswith("0"):
        digits = digits[1:]

    if len(digits) == 10 and digits[0] in "6789":
        return f"{INDIA_CC}{digits}"

    if len(digits) == 12 and digits.startswith("91") and digits[2] in "6789":
        return f"+{digits}"

    if raw.startswith("+") and 11 <= len(digits) <= 15:
        return f"+{digits}"

    logger.warning("Invalid phone skipped: %r (digits=%s)", raw, digits)
    return None


@dataclass(frozen=True)
class OutboundCallType:
    purpose: str
    call_type: str
    list_candidates: Callable[[dict], list]


OUTBOUND_CALL_TYPES: dict[str, OutboundCallType] = {
    "service_reminder": OutboundCallType(
        purpose="service_reminder",
        call_type="service",
        list_candidates=lambda data: list_vehicles_needing_service_reminder(data),
    ),
}


def log_result(vehicle: dict, purpose: str, status: str, detail: str = "") -> None:
    entry = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "vehicle_id": vehicle.get("vehicle_id", ""),
        "registration_number": vehicle.get("registration_number", ""),
        "customer_id": vehicle.get("customer_id", ""),
        "purpose": purpose,
        "status": status,
        "detail": detail,
    }
    with CALL_LOG_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


async def _wait_for_hangup(lk: api.LiveKitAPI, room_name: str, identity: str) -> None:
    while True:
        try:
            res = await lk.room.list_participants(api.ListParticipantsRequest(room=room_name))
            if identity not in [p.identity for p in res.participants]:
                return
        except Exception:
            return
        await asyncio.sleep(5)


async def place_outbound_call(
    lk: api.LiveKitAPI, entry: OutboundCallType, vehicle: dict, data: dict, semaphore: asyncio.Semaphore
) -> None:
    vehicle_id = vehicle.get("vehicle_id", "")
    customer = find_customer_by_id(data, vehicle.get("customer_id", ""))
    cust_name = customer.get("name", "Customer") if customer else "Customer"
    phone = normalise_phone(customer.get("phone", "")) if customer else None

    if not phone:
        logger.warning("Skipping %s (%s) — unusable customer phone", vehicle_id, entry.purpose)
        log_result(vehicle, entry.purpose, "skipped", "invalid phone")
        return

    room_name = f"{ROOM_PREFIX}{entry.purpose}-{vehicle_id}-{uuid.uuid4().hex[:6]}"
    metadata = json.dumps(
        {
            "mode": "outbound",
            "purpose": entry.purpose,
            "vehicle_id": vehicle_id,
            "call_type": entry.call_type,
        },
        ensure_ascii=False,
    )

    async with semaphore:
        logger.info(
            "Dialing %s for %s (%s, vehicle_id=%s) -> room %s",
            phone, cust_name, entry.purpose, vehicle_id, room_name,
        )
        try:
            await lk.agent_dispatch.create_dispatch(
                api.CreateAgentDispatchRequest(
                    agent_name=AGENT_NAME,
                    room=room_name,
                    metadata=metadata,
                )
            )

            participant = await lk.sip.create_sip_participant(
                api.CreateSIPParticipantRequest(
                    sip_trunk_id=TRUNK_ID,
                    sip_call_to=phone,
                    room_name=room_name,
                    participant_identity=f"sip_{phone.lstrip('+')}",
                    participant_name=cust_name,
                    wait_until_answered=True,
                )
            )
            logger.info("Answered: %s (room %s)", participant.participant_identity, room_name)
            await _wait_for_hangup(lk, room_name, participant.participant_identity)
            logger.info("Call ended: %s", phone)
            log_result(vehicle, entry.purpose, "completed")

        except Exception as exc:
            logger.error("Call failed for %s (%s): %s", phone, entry.purpose, exc)
            log_result(vehicle, entry.purpose, "failed", str(exc))


async def run_daily_batch(purposes: Optional[list[str]] = None, dry_run: bool = False) -> None:
    data = load_all_car_service_data()
    active_purposes = purposes or list(OUTBOUND_CALL_TYPES)

    plan: list[tuple[OutboundCallType, dict]] = []
    for purpose in active_purposes:
        entry = OUTBOUND_CALL_TYPES[purpose]
        candidates = entry.list_candidates(data)
        logger.info("%s candidates today: %d", purpose, len(candidates))
        plan.extend((entry, record) for record in candidates)

    if not plan:
        logger.info("No eligible records today. Nothing to dial.")
        return

    if dry_run:
        print("\n=== DRY RUN — no calls placed ===")
        for entry, record in plan:
            cust = find_customer_by_id(data, record.get("customer_id", ""))
            print(
                f"  [{entry.purpose}] {record.get('vehicle_id')} "
                f"{record.get('registration_number')} ({record.get('brand')} {record.get('model')}) -> "
                f"{cust.get('name') if cust else 'Unknown'} ({cust.get('phone') if cust else 'No Phone'})"
            )
        print(f"Total: {len(plan)} call(s)\n")
        return

    lk = api.LiveKitAPI(url=LIVEKIT_URL, api_key=LIVEKIT_API_KEY, api_secret=LIVEKIT_API_SECRET)
    try:
        semaphore = asyncio.Semaphore(CALL_CONCURRENCY)
        await asyncio.gather(*(place_outbound_call(lk, entry, record, data, semaphore) for entry, record in plan))
        logger.info("Batch complete: %d call(s) processed.", len(plan))
    finally:
        await lk.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(description="Outbound caller — car service reminder calls")
    parser.add_argument("--dry-run", action="store_true", help="print eligible targets, place no calls")
    parser.add_argument(
        "--purpose",
        choices=list(OUTBOUND_CALL_TYPES),
        help="run only this purpose",
    )
    args = parser.parse_args()
    purposes = [args.purpose] if args.purpose else None
    asyncio.run(run_daily_batch(purposes=purposes, dry_run=args.dry_run))


if __name__ == "__main__":
    main()