"""Telegram sender with de-duplication and rate limiting. Token/chat id come from env (never hard-coded)."""
import hashlib
import os
import time
import requests

SEV = {"safe": 0, "warning": 1, "danger": 2}


class TelegramNotifier:
    def __init__(self, token=None, chat_id=None, min_interval=600, dedupe_window=3600,
                 max_per_minute=20, clock=time.time, sender=None):
        self.token = token if token is not None else os.environ.get("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id if chat_id is not None else os.environ.get("TELEGRAM_CHAT_ID")
        self.min_interval, self.dedupe_window, self.max_per_minute = min_interval, dedupe_window, max_per_minute
        self.clock = clock
        self.sender = sender or self._post
        self.last_sent = {}       # station -> time
        self.seen = {}            # hash -> time
        self.recent = []          # timestamps of sends

    @property
    def configured(self):
        return bool(self.token and self.chat_id)

    def _post(self, text):
        r = requests.post(f"https://api.telegram.org/bot{self.token}/sendMessage",
                          json={"chat_id": self.chat_id, "text": text}, timeout=10)
        r.raise_for_status()

    def send(self, station, status, texts):
        """Returns one of: sent, duplicate, rate_limited, not_configured, error:<msg>."""
        now = self.clock()
        if not self.configured and self.sender == self._post:
            return "not_configured"
        h = hashlib.sha256(f"{station}|{status}|{texts.get('en')}".encode()).hexdigest()
        if now - self.seen.get(h, -1e18) < self.dedupe_window:
            return "duplicate"
        if status != "danger" and now - self.last_sent.get(station, -1e18) < self.min_interval:
            return "rate_limited"
        self.recent = [t for t in self.recent if now - t < 60]
        if len(self.recent) >= self.max_per_minute:
            return "rate_limited"
        icon = {"danger": "\U0001F6A8", "warning": "\u26A0\uFE0F", "safe": "\u2705"}.get(status, "")
        msg = f"{icon} {texts['en']}\n\n{texts['hi']}\n\n{texts['gu']}"
        try:
            self.sender(msg)
        except Exception as e:
            return f"error:{type(e).__name__}"
        self.seen[h] = now; self.last_sent[station] = now; self.recent.append(now)
        return "sent"
