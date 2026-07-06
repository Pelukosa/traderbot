#!/usr/bin/env python3
"""
Lee el archivo de eventos del bot y envía a WhatsApp los no notificados.
Se ejecuta como script de cron cada 5 minutos.

Usa el delivery de Hermes: el stdout se entrega al chat configurado.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

EVENTS_FILE = Path(__file__).resolve().parent.parent / "data/events.json"
# If running from ~/.hermes/scripts/, resolve to traderbot dir
if not EVENTS_FILE.exists():
    EVENTS_FILE = Path.home() / "traderbot" / "data" / "events.json"
if not EVENTS_FILE.exists():
    EVENTS_FILE = Path("/opt/data/traderbot/data/events.json")

SENT_FILE = Path(__file__).resolve().parent.parent / "data/events_sent.json"
if not SENT_FILE.parent.exists():
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


def main():
    if not EVENTS_FILE.exists():
        return  # nothing to report

    events = json.loads(EVENTS_FILE.read_text())
    if not events:
        return

    # Load sent events tracker
    sent = set()
    if SENT_FILE.exists():
        sent = set(json.loads(SENT_FILE.read_text()))

    # Find unsent events (by timestamp)
    unsent = [e for e in events if e.get("timestamp") not in sent]

    if not unsent:
        return

    # Send only the NEWEST unsent event (avoid spam)
    newest = unsent[-1]
    e = EMOJIS.get(newest["type"], "ℹ️")
    lines = [f"{e} *{newest['type'].upper()}*"]

    msg = newest.get("message", "")
    if msg:
        lines.append(f"📝 {msg}")

    price = newest.get("price")
    if price is not None:
        lines.append(f"💰 {price:.2f}€")

    pnl = newest.get("pnl")
    if pnl is not None:
        sign = "+" if pnl >= 0 else ""
        lines.append(f"📊 PnL: {sign}{pnl:.2f}€")

    balance = newest.get("balance")
    if balance is not None:
        lines.append(f"🏦 {balance:.2f}€")

    details = newest.get("details", {})
    if details:
        for k, v in details.items():
            if isinstance(v, float):
                lines.append(f"   {k}: {v:.2f}")
            else:
                lines.append(f"   {k}: {v}")

    t = newest.get("timestamp", "")
    if t:
        lines.append(f"🕐 {t[11:16] if len(t) > 16 else t}")

    # Mark as sent
    sent.add(newest["timestamp"])
    SENT_FILE.write_text(json.dumps(list(sent)))

    # Print for delivery via cron
    print("\n".join(lines))


if __name__ == "__main__":
    main()
