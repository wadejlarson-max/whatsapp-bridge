import json
import os
import time
from pathlib import Path

import requests
from openai import OpenAI

BASE = "https://api.whatsapp.com/agent/v1"
TOKEN = os.environ["WHATSAPP_API_KEY"]
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
STATE = Path(os.environ.get("STATE_PATH", "/data/state.json"))

client = OpenAI(timeout=60, max_retries=2)
session = requests.Session()
session.headers["Authorization"] = f"Bearer {TOKEN}"

state = (
    json.loads(STATE.read_text())
    if STATE.exists()
    else {"offset": 0, "seen": [], "history": []}
)


def save():
    STATE.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(state))
    temporary.replace(STATE)


def send(recipient, text):
    time.sleep(6)
    response = session.post(
        f"{BASE}/messages",
        json={
            "messaging_product": "whatsapp",
            "to": recipient,
            "type": "text",
            "text": {"body": text[:4000]},
        },
        timeout=60,
    )
    if not response.ok:
        raise RuntimeError(f"WhatsApp send HTTP {response.status_code}")


def handle(message):
    message_id = message["id"]
    if message_id in state["seen"]:
        return

    recipient = message["from"]
    if not recipient.startswith("user:"):
        return

    if message.get("type") == "reaction":
        return

    if message.get("type") != "text":
        reply = "This first version supports text messages only."
    else:
        text = message["text"]["body"]
        if text.strip().lower() == "/reset":
            state["history"] = []
            reply = "Our conversation context has been cleared."
        else:
            history = state["history"][-20:]
            history.append({"role": "user", "content": text})
            result = client.responses.create(
                model=MODEL,
                instructions=(
                    "You are Lena, Wade's AI assistant on WhatsApp. "
                    "Be warm, candid, clear and concise. "
                    "You have only the context supplied in this conversation. "
                    "Do not claim access to his ChatGPT history, other chats, "
                    "email, web browsing or tools."
                ),
                input=history,
                max_output_tokens=900,
                store=False,
            )
            reply = result.output_text.strip()
            if not reply:
                raise RuntimeError("OpenAI returned no text")
            reply = reply[:4000]
            history.append({"role": "assistant", "content": reply})
            state["history"] = history[-20:]

    # Record before sending to avoid duplicate replies after a restart.
    # An uncertain send failure is not automatically retried.
    state["seen"] = (state["seen"] + [message_id])[-2000:]
    save()
    send(recipient, reply)
    print("Reply sent", flush=True)


def main():
    save()
    print("WhatsApp bridge started", flush=True)
    while True:
        started = time.monotonic()
        try:
            response = session.get(
                f"{BASE}/updates",
                params={
                    "offset": state["offset"],
                    "limit": 50,
                    "timeout": 25,
                },
                timeout=40,
            )
            if response.status_code == 204:
                continue
            if not response.ok:
                raise RuntimeError(
                    f"WhatsApp poll HTTP {response.status_code}"
                )
            data = response.json()
            for entry in data.get("entry", []):
                for change in entry.get("changes", []):
                    value = change.get("value", {})
                    for message in value.get("messages", []):
                        handle(message)
            state["offset"] = data["next_offset"]
            save()
        except Exception as error:
            # Log the error type, never keys or message contents.
            print(
                f"Bridge error: {type(error).__name__}",
                flush=True,
            )
            time.sleep(15)
        finally:
            remaining = 5 - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
