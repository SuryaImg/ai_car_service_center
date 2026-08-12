"""
token_server.py — Lightweight HTTP server for LiveKit access tokens.

Serves tokens at GET /token?user_id=xxx so a browser/test client can join a
LiveKit room and talk to the unified agent (common.UNIFIED_AGENT_NAME).
"""

import json
import os
import uuid
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

from dotenv import load_dotenv
from livekit import api

from common import UNIFIED_AGENT_NAME

load_dotenv()

LIVEKIT_API_KEY = os.environ.get("LIVEKIT_API_KEY", "")
LIVEKIT_API_SECRET = os.environ.get("LIVEKIT_API_SECRET", "")
LIVEKIT_URL = os.environ.get("LIVEKIT_URL", "")
PORT = 8080


class TokenHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/token":
            self._handle_token(parsed)
        else:
            self.send_response(404)
            self.end_headers()

    def _handle_token(self, parsed):
        params = parse_qs(parsed.query)
        user_id = params.get("user_id", ["test-caller"])[0]

        room_name = f"reception-{user_id}-{uuid.uuid4().hex[:8]}"

        token = (
            api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
            .with_identity(f"caller-{user_id}")
            .with_name(f"Caller {user_id}")
            .with_grants(
                api.VideoGrants(
                    room_join=True,
                    room=room_name,
                    can_publish=True,
                    can_subscribe=True,
                )
            )
            .with_room_config(
                api.RoomConfiguration(
                    agents=[
                        api.RoomAgentDispatch(
                            agent_name=UNIFIED_AGENT_NAME,
                            metadata=json.dumps({"user_id": user_id}),
                        )
                    ],
                )
            )
            .to_jwt()
        )

        body = json.dumps(
            {
                "token": token,
                "url": LIVEKIT_URL,
                "room": room_name,
                "identity": f"caller-{user_id}",
                "agent_name": UNIFIED_AGENT_NAME,
            }
        ).encode()

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        print(f"[TokenServer] {args[0]}")


if __name__ == "__main__":
    print(f"Token server running at http://localhost:{PORT}")
    print(f"  Open http://localhost:{PORT} to test the agent")
    print(f"  Token endpoint: http://localhost:{PORT}/token?user_id=<phone>")
    print(f"  Every call goes to the single unified agent: {UNIFIED_AGENT_NAME}")
    HTTPServer(("0.0.0.0", PORT), TokenHandler).serve_forever()
