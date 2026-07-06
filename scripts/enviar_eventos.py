#!/usr/bin/env python3
"""
Lee eventos del bot y los envía por Gotify.
Se ejecuta como cron cada 3 minutos.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

GOTIFY_URL = "http://192.168.1.6:8492"
GOTIFY_TOKEN = "AnKlAMgG66v.vmf"
EVENTS_FILE = Path("/opt/data/traderbot/data/events.json")
SENT_FILE = Path("/opt/data/traderbot/data/events_sent.json")

EMOJIS = {
    "buy": "🟢",
    "sell": "🔴",
    "sl": "⛔",
    "trailing": "📈",
    "error": "🚨",
    "startup": "🤖",
    "shutdown": "💤",
    "signal": "📊",
    "heartbeat": "💓",
}

PRIORITIES = {
    "buy": 5,
    "sell": 5,
    "sl": 8,
    "error": 10,
    "startup": 3,
    "shutdown": 3,
    "trailing": 4,
    "heartbeat": 2,
}


def send_gotify(title: str, message: str, priority: int = 5) -> bool:
    payload = json.dumps({
        "title": title,
        "message": message,
        "priority": priority,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{GOTIFY_URL}/message?token={GOTIFY_TOKEN}",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(req, timeout=10)
        return True
    except Exception as e:
        print(f"Gotify error: {e}")
        return False


def main():
    if not EVENTS_FILE.exists():
        return

    events = json.loads(EVENTS_FILE.read_text())
    if not events:
        return

    sent = set()
    if SENT_FILE.exists():
        sent = set(json.loads(SENT_FILE.read_text()))

    unsent = [e for e in events if e.get("timestamp") not in sent]
    if not unsent:
        return

    # Send ALL unsent events (up to 5 per tick to avoid spam)
    for event in unsent[-5:]:
        etype = event["type"]
        emoji = EMOJIS.get(etype, "ℹ️")
        priority = PRIORITIES.get(etype, 5)

        lines = []

        msg = event.get("message", "")
        if msg:
            lines.append(msg)

        price = event.get("price")
        if price is not None:
            lines.append(f"💰 {price:.2f}€")

        pnl = event.get("pnl")
        if pnl is not None:
            sign = "+" if pnl >= 0 else ""
            lines.append(f"📊 PnL: {sign}{pnl:.2f}€")

        balance = event.get("balance")
        if balance is not None:
            lines.append(f"🏦 {balance:.2f}€")

        t = event.get("timestamp", "")
        if t:
            lines.append(f"🕐 {t[11:16] if len(t) > 16 else t}")

        title = f"{emoji} {etype.upper()}"
        message = "\n".join(lines) if lines else etype

        if send_gotify(title, message, priority):
            sent.add(event["timestamp"])
        else:
            # Don't mark as sent if it failed — will retry next tick
            pass

    SENT_FILE.write_text(json.dumps(list(sent)))


if __name__ == "__main__":
    main()
