"""
Notificaciones vía Gotify para eventos del bot.

Se llama desde el execution manager en cada compra/venta/error/startup.
Además escribe a data/events.json para el cron de respaldo.
"""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

# ── Gotify ──
GOTIFY_URL = "http://192.168.1.6:8492"
GOTIFY_TOKEN = "AnKlAMgG66v.vmf"

# ── Event file (backup / cron) ──
EVENTS_FILE = Path(__file__).parent.parent / "data/events.json"

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


@dataclass
class BotEvent:
    type: str  # buy | sell | error | startup | shutdown | sl | trailing
    message: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    price: float | None = None
    pnl: float | None = None
    balance: float | None = None
    details: dict[str, Any] = field(default_factory=dict)


def _save_event(event: BotEvent):
    """Save event to local file for history."""
    EVENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        events = []
        if EVENTS_FILE.exists():
            events = json.loads(EVENTS_FILE.read_text())
        events.append(asdict(event))
        EVENTS_FILE.write_text(json.dumps(events[-100:], indent=2))
    except Exception:
        pass


def _send_gotify(title: str, message: str, priority: int = 5) -> bool:
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
        logger.warning("Gotify send failed: {}", e)
        return False


def _format_event(event: BotEvent) -> tuple[str, str, int]:
    """Returns (title, message, priority)."""
    etype = event.type
    emoji = EMOJIS.get(etype, "ℹ️")
    priority = PRIORITIES.get(etype, 5)
    title = f"{emoji} {etype.upper()}"

    lines = []
    if event.message:
        lines.append(event.message)
    if event.price is not None:
        lines.append(f"💰 {event.price:.2f}€")
    if event.pnl is not None:
        sign = "+" if event.pnl >= 0 else ""
        lines.append(f"📊 PnL: {sign}{event.pnl:.2f}€")
    if event.balance is not None:
        lines.append(f"🏦 {event.balance:.2f}€")
    if event.details:
        for k, v in event.details.items():
            if isinstance(v, float):
                lines.append(f"   {k}: {v:.2f}")
            else:
                lines.append(f"   {k}: {v}")
    t = event.timestamp
    if t:
        lines.append(f"🕐 {t[11:16] if len(t) > 16 else t}")

    return title, "\n".join(lines) if lines else etype, priority


def notify(event: BotEvent) -> bool:
    """Send to Gotify AND save to events file."""
    _save_event(event)

    title, message, priority = _format_event(event)
    ok = _send_gotify(title, message, priority)

    logger.info("NOTIFY: {} | {} | priority={} | gotify={}",
                title, message.replace("\n", " | "), priority, ok)

    return ok


# ── Helper functions ──


def notify_buy(price: float, balance: float, details: dict | None = None):
    return notify(BotEvent(
        type="buy",
        message="Compra ejecutada",
        price=price,
        balance=balance,
        details=details or {},
    ))


def notify_sell(price: float, pnl: float, balance: float, reason: str, details: dict | None = None):
    return notify(BotEvent(
        type="sell",
        message=f"Venta ejecutada — {reason}",
        price=price,
        pnl=pnl,
        balance=balance,
        details=details or {},
    ))


def notify_sl(price: float, pnl: float, balance: float):
    return notify(BotEvent(
        type="sl",
        message="Stop-loss alcanzado",
        price=price,
        pnl=pnl,
        balance=balance,
    ))


def notify_trailing(price: float, sl_price: float, gain_pct: float):
    return notify(BotEvent(
        type="trailing",
        message=f"Trailing ajustado — SL en {sl_price:.2f}€ (+{gain_pct:.2f}%)",
        price=price,
    ))


def notify_error(error_msg: str, details: dict | None = None):
    return notify(BotEvent(
        type="error",
        message=error_msg,
        details=details or {},
    ))


def notify_startup(balance: float):
    return notify(BotEvent(
        type="startup",
        message="Bot iniciado — modo LIVE",
        balance=balance,
    ))


def notify_shutdown():
    return notify(BotEvent(
        type="shutdown",
        message="Bot detenido",
    ))


def notify_heartbeat(balance: float, price: float, in_position: bool):
    """Daily heartbeat so you know it's alive."""
    pos = "EN POSICIÓN" if in_position else "ESPERANDO"
    return notify(BotEvent(
        type="heartbeat",
        message=f"{pos} | 💰{price:.0f}€ | 🏦{balance:.0f}€",
        price=price,
        balance=balance,
    ))
