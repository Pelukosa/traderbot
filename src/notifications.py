"""
Notificaciones vía WhatsApp para eventos del bot.

Usa el webhook de Hermes para enviar mensajes a través de WhatsApp.
Se llama desde el execution manager en cada compra/venta/error/startup.
"""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

# ── Config ──
# Se puede sobrescribir con variables de entorno
HERMES_CLI = "/opt/data/home/.local/bin/hermes"
NOTIFY_CHAT = "whatsapp:89064904589410@lid"  # DM directo a Alejandro
NOTIFY_ENABLED = True

# Fichero de eventos para persistencia
EVENTS_FILE = Path(__file__).parent.parent / "data/events.json"


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
        # Keep last 100
        EVENTS_FILE.write_text(json.dumps(events[-100:], indent=2))
    except Exception:
        pass


def _emoji(event_type: str) -> str:
    return {
        "buy": "🟢",
        "sell": "🔴",
        "sl": "⛔",
        "trailing": "📈",
        "error": "🚨",
        "startup": "🤖",
        "shutdown": "💤",
        "signal": "📊",
        "heartbeat": "💓",
    }.get(event_type, "ℹ️")


def format_message(event: BotEvent) -> str:
    """Format a BotEvent as a concise WhatsApp message."""
    e = _emoji(event.type)
    lines = [f"{e} *{event.type.upper()}*"]

    if event.message:
        lines.append(f"📝 {event.message}")

    if event.price is not None:
        lines.append(f"💰 Precio: {event.price:.2f}€")

    if event.pnl is not None:
        sign = "+" if event.pnl >= 0 else ""
        lines.append(f"📊 PnL: {sign}{event.pnl:.2f}€")

    if event.balance is not None:
        lines.append(f"🏦 Balance: {event.balance:.2f}€")

    if event.details:
        for k, v in event.details.items():
            if isinstance(v, float):
                lines.append(f"   {k}: {v:.2f}")
            else:
                lines.append(f"   {k}: {v}")

    t = datetime.fromisoformat(event.timestamp)
    lines.append(f"🕐 {t.strftime('%H:%M:%S')}")

    return "\n".join(lines)


def send_whatsapp(text: str) -> bool:
    """Send a WhatsApp message using the Hermes send_message tool."""
    if not NOTIFY_ENABLED:
        logger.debug("Notifications disabled — would send: {}", text[:80])
        return False

    try:
        # Import Hermes send_message tool dynamically
        result = subprocess.run(
            [HERMES_CLI, "send_message", NOTIFY_CHAT, text],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            logger.info("WhatsApp notification sent")
            return True
        else:
            logger.warning("WhatsApp send failed: {}", result.stderr[:300])
            return False
    except FileNotFoundError:
        logger.warning("Hermes CLI not found — notification not sent")
        return False
    except Exception as e:
        logger.warning("Failed to send notification: {}", e)
        return False


def notify(event: BotEvent) -> bool:
    """Send a notification and save the event."""
    _save_event(event)

    msg = format_message(event)
    logger.info("NOTIFY: {}", msg.replace("\n", " | "))

    return send_whatsapp(msg)


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
    pos = "🟢 EN POSICIÓN" if in_position else "⚪ ESPERANDO SEÑAL"
    return notify(BotEvent(
        type="heartbeat",
        message=f"{pos}",
        price=price,
        balance=balance,
    ))
