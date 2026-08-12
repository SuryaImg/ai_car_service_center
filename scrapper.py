"""
scrapper.py — Live Playwright scraper for Hospital Dashboard
"""

import json
import os
import time

_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(_DIR, "hospital_cache.json")

HOSPITAL_URL = "https://96e7-2406-b400-75-20fa-dced-c4f0-5c94-c4eb.ngrok-free.app/"

def capture_api_data(headless: bool = True, attempt: int = 1) -> dict:
    """
    Launch Playwright, navigate to the hospital URL, intercept all JSON
    API responses, and return the merged data.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright not installed")
        return load_cache() or {}

    api_responses: list[dict] = []

    def handle_response(response):
        try:
            if "json" in response.headers.get("content-type", ""):
                body = response.json()
                if body:
                    api_responses.append({"url": response.url, "data": body})
        except Exception:
            pass

    def block_resources(route, request):
        if request.resource_type in ("image", "stylesheet", "font", "media"):
            route.abort()
        else:
            route.continue_()

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=headless,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-gpu",
                    "--blink-settings=imagesEnabled=false",
                ],
            )
            context = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                )
            )
            page = context.new_page()
            page.route("**/*", block_resources)
            page.on("response", handle_response)

            page.goto(HOSPITAL_URL, wait_until="domcontentloaded", timeout=30_000)
            
            try:
                page.wait_for_load_state("networkidle", timeout=15_000)
            except Exception:
                pass
            
            page.wait_for_timeout(2_000)
            browser.close()

    except Exception as e:
        print(f"Playwright error (attempt {attempt}): {e}")
        if attempt < 2:
            return capture_api_data(headless=headless, attempt=2)
        cached = load_cache()
        return cached if cached else {}

    # Merge intercepted JSON data
    merged = {}
    for entry in api_responses:
        data = entry.get("data", {})
        merged.update(data)

    if merged:
        save_cache(merged)
        return merged
    
    cached = load_cache()
    return cached if cached else {}

def save_cache(data: dict) -> None:
    payload = {
        "data": data,
        "saved_at": time.time(),
        "saved_at_human": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(CACHE_FILE, "w") as f:
        json.dump(payload, f, indent=2, default=str)

def load_cache(max_age_seconds: float = 7200.0) -> dict | None:
    if not os.path.exists(CACHE_FILE):
        return None
    try:
        with open(CACHE_FILE) as f:
            payload = json.load(f)
        age = time.time() - payload.get("saved_at", 0)
        if age > max_age_seconds:
            return None
        return payload.get("data")
    except Exception:
        return None

def fetch_hospital_data() -> dict:
    return capture_api_data(headless=True)

if __name__ == "__main__":
    data = fetch_hospital_data()
    if data:
        print("Successfully fetched live data via Playwright.")
        print(f"Found keys: {list(data.keys())}")
    else:
        print("Failed to fetch data.")
