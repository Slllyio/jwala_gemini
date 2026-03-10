"""
Notifier — Slack / Discord / Email Alerts
==========================================

Sends structured notifications for pipeline events:
  - Run completion (success/failure summary)
  - New encroachment alerts above threshold
  - Step failure with error context

Usage:
    from src.monitoring import Notifier

    notifier = Notifier.from_config("config.yaml")
    notifier.send_run_complete(report_dict)
    notifier.send_encroachment_alert(n_alerts=12, total_ha=45.6)
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Optional

import yaml

log = logging.getLogger(__name__)

# Rate limit: max 1 message per channel per 30s
_last_sent: dict[str, float] = {}
_RATE_LIMIT_S = 30


class Notifier:
    """
    Multi-channel notifier supporting Slack, Discord, and email.

    Channels are configured via config.yaml or environment variables.
    """

    def __init__(
        self,
        slack_webhook: Optional[str] = None,
        discord_webhook: Optional[str] = None,
        email_config: Optional[dict] = None,
        enabled: bool = True,
    ):
        self.slack_webhook = slack_webhook or os.environ.get("VS_SLACK_WEBHOOK")
        self.discord_webhook = discord_webhook or os.environ.get("VS_DISCORD_WEBHOOK")
        self.email_config = email_config
        self.enabled = enabled

    @classmethod
    def from_config(cls, config_path: str = "config.yaml") -> "Notifier":
        """Create a Notifier from config.yaml."""
        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            nc = cfg.get("notifications", {})
            return cls(
                slack_webhook=nc.get("slack_webhook"),
                discord_webhook=nc.get("discord_webhook"),
                email_config=nc.get("email"),
                enabled=nc.get("enabled", True),
            )
        except FileNotFoundError:
            log.warning(f"Config not found: {config_path}, notifications disabled")
            return cls(enabled=False)

    # ── Public API ───────────────────────────────────────────

    def send_run_complete(self, report: dict):
        """Send a pipeline run completion notification."""
        status = report.get("status", "unknown")
        emoji = "✅" if status == "success" else "❌"
        name = report.get("run_name", "unknown")
        duration = report.get("total_seconds", 0)
        n_steps = report.get("steps_total", 0)
        n_fail = report.get("steps_failed", 0)

        text = (
            f"{emoji} *Pipeline: {name}*\n"
            f"Status: {status.upper()} | {n_steps} steps "
            f"({n_fail} failed) | {duration:.1f}s"
        )

        # Add failed step details
        if n_fail > 0:
            failed = [
                s for s in report.get("steps", [])
                if s.get("status") == "failed"
            ]
            for s in failed[:3]:  # max 3
                text += f"\n  💥 `{s['name']}`: {s.get('error', 'unknown')}"

        self._send_all(text, channel="pipeline")

    def send_encroachment_alert(
        self,
        n_alerts: int,
        total_ha: float,
        sub_range: Optional[str] = None,
        detection_date: Optional[str] = None,
    ):
        """Send alert when new encroachment detections exceed threshold."""
        location = f" in *{sub_range}*" if sub_range else ""
        date_str = f" ({detection_date})" if detection_date else ""

        text = (
            f"🚨 *Encroachment Alert{location}*{date_str}\n"
            f"{n_alerts} new encroachment polygons detected | "
            f"{total_ha:.2f} ha total area\n"
            f"Review: `SELECT * FROM alerts_log WHERE change_type = "
            f"'Encroachment' ORDER BY ingested_at DESC LIMIT {n_alerts};`"
        )

        self._send_all(text, channel="alerts")

    def send_error(
        self,
        run_name: str,
        step_name: str,
        error: str,
        trace: Optional[str] = None,
    ):
        """Send an error notification for a failed pipeline step."""
        text = (
            f"💥 *Pipeline Error: {run_name}*\n"
            f"Step `{step_name}` failed:\n"
            f"```{error}```"
        )
        if trace:
            # Truncate trace to last 500 chars
            short_trace = trace[-500:] if len(trace) > 500 else trace
            text += f"\n```{short_trace}```"

        self._send_all(text, channel="errors")

    def send_model_registered(
        self,
        version: str,
        metrics: Optional[dict] = None,
    ):
        """Notify when a new model version is registered."""
        text = f"📦 *Model Registered: {version}*"
        if metrics:
            for k, v in list(metrics.items())[:5]:
                text += f"\n  {k}: {v}"

        self._send_all(text, channel="models")

    # ── Transport layer ──────────────────────────────────────

    def _send_all(self, text: str, channel: str = "general"):
        """Send message to all configured channels with rate limiting."""
        if not self.enabled:
            log.debug(f"[notifier:disabled] {text[:80]}...")
            return

        # Rate limiting
        key = f"{channel}"
        now = time.time()
        if key in _last_sent and (now - _last_sent[key]) < _RATE_LIMIT_S:
            log.debug(f"[notifier:rate-limited] {channel}")
            return
        _last_sent[key] = now

        if self.slack_webhook:
            self._send_slack(text)
        if self.discord_webhook:
            self._send_discord(text)
        if self.email_config:
            self._send_email(text, channel)

        if not any([self.slack_webhook, self.discord_webhook, self.email_config]):
            log.info(f"[notifier:console] {text}")

    def _send_slack(self, text: str):
        """Post to Slack via incoming webhook."""
        try:
            import requests
            payload = {
                "text": text,
                "unfurl_links": False,
                "unfurl_media": False,
            }
            resp = requests.post(
                self.slack_webhook,
                json=payload,
                timeout=10,
            )
            if resp.status_code != 200:
                log.warning(f"Slack webhook returned {resp.status_code}")
        except Exception as e:
            log.warning(f"Slack notification failed: {e}")

    def _send_discord(self, text: str):
        """Post to Discord via webhook."""
        try:
            import requests
            # Discord uses 'content' instead of 'text'
            # Convert Slack markdown (*bold*) to Discord (**bold**)
            discord_text = text.replace("*", "**")
            payload = {"content": discord_text}
            resp = requests.post(
                self.discord_webhook,
                json=payload,
                timeout=10,
            )
            if resp.status_code not in (200, 204):
                log.warning(f"Discord webhook returned {resp.status_code}")
        except Exception as e:
            log.warning(f"Discord notification failed: {e}")

    def _send_email(self, text: str, channel: str):
        """Send via SMTP."""
        try:
            import smtplib
            from email.mime.text import MIMEText

            ec = self.email_config
            if not ec:
                return

            to_addr = ec.get("to_addr", "")
            from_addr = ec.get("from_addr", "")
            if not to_addr or not from_addr:
                log.warning("Email notification skipped: from_addr or to_addr not configured")
                return

            msg = MIMEText(text)
            msg["Subject"] = f"[Van Suraksha] {channel.title()} Notification"
            msg["From"] = from_addr
            msg["To"] = to_addr

            with smtplib.SMTP(
                ec.get("smtp_host", "localhost"),
                ec.get("smtp_port", 587),
            ) as server:
                if ec.get("use_tls", True):
                    server.starttls()
                if ec.get("username"):
                    server.login(ec["username"], ec.get("password", ""))
                server.send_message(msg)

            log.info(f"Email sent to {msg['To']}")
        except Exception as e:
            log.warning(f"Email notification failed: {e}")
