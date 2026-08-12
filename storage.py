"""
storage.py — Redis-backed onboarding state + call transcript + seller URL lookup.

Key schema:
  onboarding:{user_id}          → JSON blob of OnboardingState fields
                                   TTL: REDIS_STATE_TTL_DAYS days (default 30)
  seller_url:{phone}            → seller-specific Flipkart dashboard URL
                                   TTL: 7 days (set externally by the scraper/ingestion pipeline)
  transcript:{user_id}:{ts}     → JSON blob of CallTranscript
                                   TTL: 90 days

Fallback: if Redis is unreachable the module silently uses an in-process
dict so the agent never crashes due to storage issues.

.env vars:
  REDIS_URL   redis://localhost:6379      (default)
              rediss://user:pass@host:6380  (TLS)

          
"""

import asyncio
import json
import logging
import os
import time
from typing import Optional

from dotenv import load_dotenv
from tools import AppointmentState, CallTranscript

load_dotenv()

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
REDIS_URL                = os.environ.get("REDIS_URL", "redis://localhost:6379")
REDIS_STATE_TTL_SEC      = int(os.environ.get("REDIS_STATE_TTL_DAYS", "30")) * 86400
REDIS_TRANSCRIPT_TTL_SEC = 90 * 86400   # 90 days
REDIS_SELLER_URL_TTL_SEC = 7 * 86400    # 7 days
KEY_PREFIX               = "appointment:"
SELLER_URL_PREFIX        = "seller_url:"
TRANSCRIPT_PREFIX        = "transcript:"

# ── Connection singleton ──────────────────────────────────────────────────────
_redis = None
_use_memory: bool = False
_memory_store: dict[str, AppointmentState] = {}
_memory_transcripts: dict[str, dict] = {}
_memory_seller_urls: dict[str, str] = {}

# FIX-10: Never create asyncio.Lock at module level.
# It is created lazily on first use, inside a running event loop.
_connect_lock: Optional[asyncio.Lock] = None
_connect_lock_loop: Optional[asyncio.AbstractEventLoop] = None


def _get_connect_lock() -> asyncio.Lock:
    """
    Return the connection lock, creating or recreating it as needed.

    FIX-10: asyncio.Lock must be created inside a running event loop.
    We track which loop the lock was created for and recreate it if the
    loop has changed (common in tests or worker process restarts).
    """
    global _connect_lock, _connect_lock_loop

    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop yet — caller is not inside an async context.
        if _connect_lock is None:
            _connect_lock = asyncio.Lock()
        return _connect_lock

    # Recreate if lock doesn't exist or belongs to a different loop
    if _connect_lock is None or _connect_lock_loop is not current_loop:
        _connect_lock = asyncio.Lock()
        _connect_lock_loop = current_loop

    return _connect_lock


def _reset_redis_state() -> None:
    """
    Reset all Redis connection state for a new worker process.

    Called from prewarm() so each livekit-agents worker subprocess starts
    with a clean slate.  Without this, the Redis asyncio client created in
    one process gets reused in another process with a *different* event loop,
    producing:
        'Future <Future pending> attached to a different loop'
    which silently drops every Redis write (transcripts, state saves, etc.)
    """
    global _redis, _use_memory, _connect_lock, _connect_lock_loop
    _redis = None
    _use_memory = False
    _connect_lock = None
    _connect_lock_loop = None
    logger.debug("Redis state reset for new worker process")


async def _get_redis():
    """Return a live Redis client, or None if unavailable."""
    global _redis, _use_memory

    if _use_memory:
        return None
    if _redis is not None:
        return _redis

    async with _get_connect_lock():
        if _redis is not None:
            return _redis
        if _use_memory:
            return None

        try:
            import redis.asyncio as aioredis

            client = aioredis.from_url(
                REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
                retry_on_timeout=False,
            )
            await asyncio.wait_for(client.ping(), timeout=2.0)
            _redis = client
            logger.info("Redis connected: %s", REDIS_URL)
            return _redis

        except Exception as e:
            logger.warning(
                "Redis unavailable (%s) — falling back to in-memory storage", e
            )
            _use_memory = True
            return None


def _key(user_id: str) -> str:
    return f"{KEY_PREFIX}{user_id}"


def _seller_url_key(phone: str) -> str:
    # Normalize: strip leading '+' and spaces
    phone = phone.strip().lstrip("+")
    return f"{SELLER_URL_PREFIX}{phone}"


def _transcript_key(user_id: str, call_ts: str) -> str:
    return f"{TRANSCRIPT_PREFIX}{user_id}:{call_ts}"


def _state_to_dict(state: AppointmentState) -> dict:
    return state.model_dump()


def _dict_to_state(data: dict) -> AppointmentState:
    return AppointmentState(**data)


# ── Onboarding State ──────────────────────────────────────────────────────────

async def lock_appointment_slot(doctor_name: str, date: str, time_slot: str, user_id: str) -> bool:
    """
    Attempts to lock a specific appointment slot globally using Redis SETNX.
    Returns True if successfully locked, False if already taken.
    """
    r = await _get_redis()
    # Normalize the key to prevent case/spacing mismatches
    doc = doctor_name.strip().lower().replace(" ", "_")
    d = date.strip().lower().replace(" ", "_")
    t = time_slot.strip().lower().replace(" ", "_")
    
    key = f"booked_slot:{doc}:{d}:{t}"
    
    if r is None:
        # Memory fallback (not safe across processes, but better than crashing)
        if key in _memory_store:
            return False
        _memory_store[key] = user_id
        logger.info("Locked slot in memory: %s for user=%s", key, user_id)
        return True

    try:
        # SET NX (only set if not exists), EX (expire in 90 days)
        # Returns True if set, None/False if it already existed
        acquired = await r.set(key, user_id, nx=True, ex=REDIS_STATE_TTL_SEC)
        if acquired:
            logger.info("Locked slot in Redis: %s for user=%s", key, user_id)
            return True
        else:
            logger.info("Slot already taken in Redis: %s (requested by user=%s)", key, user_id)
            return False
    except Exception as e:
        logger.warning("Redis SETNX failed for slot locking (%s) — allowing to prevent total failure", e)
        return True

async def get_appointment_state(user_id: str) -> Optional[AppointmentState]:
    """Fetch patient appointment state. Returns None if no prior state exists."""
    r = await _get_redis()
    if r is None:
        return _memory_store.get(user_id)

    try:
        raw = await r.get(_key(user_id))
        if raw is None:
            return None
        data = json.loads(raw)
        state = _dict_to_state(data)
        logger.debug(
            "Loaded state from Redis for user=%s step=%s",
            user_id,
            state.current_step(),
        )
        return state
    except Exception as e:
        logger.warning("Redis GET failed (%s) — memory fallback", e)
        return _memory_store.get(user_id)


async def save_appointment_state(user_id: str, state: AppointmentState) -> None:
    """Upsert appointment state."""
    r = await _get_redis()
    if r is None:
        _memory_store[user_id] = state
        logger.info(
            "Saved state in memory for user=%s step=%s",
            user_id,
            state.current_step(),
        )
        return

    try:
        payload = json.dumps(_state_to_dict(state))
        await r.set(_key(user_id), payload, ex=REDIS_STATE_TTL_SEC)
        logger.info(
            "Saved state in Redis for user=%s step=%s",
            user_id,
            state.current_step(),
        )
    except Exception as e:
        logger.warning("Redis SET failed (%s) — falling back to memory", e)
        _memory_store[user_id] = state


# ── Seller URL Lookup ─────────────────────────────────────────────────────────

async def get_seller_url(phone: str) -> Optional[str]:
    """
    Fetch seller-specific Flipkart dashboard URL from Redis.

    The external ingestion pipeline stores URLs as:
        SET seller_url:{phone_digits}  <url>  EX 604800

    Phone is normalized: leading '+' and country code variations are tried.
    Returns None if not found.
    """
    r = await _get_redis()
    phone_normalized = phone.strip().lstrip("+")

    # Try multiple key variations to handle country-code differences
    variants = [
        phone_normalized,                           # 919876543210
        phone_normalized[-10:],                     # 9876543210  (last 10)
        "91" + phone_normalized[-10:],              # 919876543210
    ]
    variants = list(dict.fromkeys(variants))        # dedup, preserve order

    if r is None:
        # Memory fallback
        for v in variants:
            url = _memory_seller_urls.get(v)
            if url:
                logger.info("Seller URL (memory) for phone=%s → %s", phone, url)
                return url
        logger.info("No seller URL found in memory for phone=%s", phone)
        return None

    try:
        for v in variants:
            url = await r.get(_seller_url_key(v))
            if url:
                logger.info("Seller URL from Redis for phone=%s → %s", phone, url)
                return url
        logger.info("No seller URL in Redis for phone=%s (tried %s)", phone, variants)
        return None
    except Exception as e:
        logger.warning("Redis seller_url GET failed (%s)", e)
        return None


async def set_seller_url(phone: str, url: str) -> None:
    """
    Store a seller's dashboard URL in Redis.
    Useful for testing / manual seeding without an external pipeline.
    """
    r = await _get_redis()
    phone_normalized = phone.strip().lstrip("+")

    if r is None:
        _memory_seller_urls[phone_normalized] = url
        logger.info("Seller URL saved in memory: %s → %s", phone_normalized, url)
        return

    try:
        await r.set(
            _seller_url_key(phone_normalized),
            url,
            ex=REDIS_SELLER_URL_TTL_SEC,
        )
        logger.info("Seller URL saved in Redis: %s → %s", phone_normalized, url)
    except Exception as e:
        logger.warning("Redis seller_url SET failed (%s)", e)
        _memory_seller_urls[phone_normalized] = url


# ── Call Transcript ───────────────────────────────────────────────────────────

async def save_call_transcript(user_id: str, transcript: "CallTranscript") -> str:
    """
    Persist the call transcript to Redis.
    Returns the Redis key used (useful for debugging).
    """
    r = await _get_redis()
    call_ts = transcript.call_start_time or str(int(time.time()))
    key = _transcript_key(user_id, call_ts)

    payload = json.dumps(transcript.model_dump(), default=str)

    if r is None:
        _memory_transcripts[key] = json.loads(payload)
        logger.info("Transcript saved in memory for user=%s key=%s", user_id, key)
        return key

    try:
        await r.set(key, payload, ex=REDIS_TRANSCRIPT_TTL_SEC)
        logger.info("Transcript saved in Redis for user=%s key=%s", user_id, key)
    except Exception as e:
        logger.warning("Redis transcript SET failed (%s) — memory fallback", e)
        _memory_transcripts[key] = json.loads(payload)

    return key


async def get_call_transcript(user_id: str, call_ts: str) -> Optional[dict]:
    """Retrieve a call transcript by user_id and call timestamp."""
    r = await _get_redis()
    key = _transcript_key(user_id, call_ts)

    if r is None:
        return _memory_transcripts.get(key)

    try:
        raw = await r.get(key)
        if raw is None:
            return _memory_transcripts.get(key)
        return json.loads(raw)
    except Exception as e:
        logger.warning("Redis transcript GET failed (%s)", e)
        return _memory_transcripts.get(key)


async def get_latest_transcript(user_id: str) -> Optional[dict]:
    """
    Return the most recent call transcript for a user by scanning keys.
    Falls back to memory store if Redis unavailable.
    """
    r = await _get_redis()
    pattern = _transcript_key(user_id, "*")

    if r is None:
        # Scan memory
        matching = {k: v for k, v in _memory_transcripts.items() if k.startswith(f"{TRANSCRIPT_PREFIX}{user_id}:")}
        if not matching:
            return None
        latest_key = max(matching.keys())
        return matching[latest_key]

    try:
        keys = []
        async for key in r.scan_iter(pattern):
            keys.append(key)
        if not keys:
            return None
        latest_key = max(keys)
        raw = await r.get(latest_key)
        return json.loads(raw) if raw else None
    except Exception as e:
        logger.warning("Redis transcript scan failed (%s)", e)
        return None


# ── Connection teardown ───────────────────────────────────────────────────────

async def close_pool() -> None:
    """Gracefully close the Redis connection (called on worker shutdown)."""
    global _redis
    if _redis is not None:
        try:
            await _redis.aclose()
        except Exception:
            pass
        _redis = None