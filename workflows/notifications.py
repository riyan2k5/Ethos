"""
Notification handlers for Prefect workflow pipeline.
Supports Discord notifications via webhook.
"""

import os
from typing import Dict, Any, Optional
from datetime import datetime
import httpx
from prefect import get_run_logger


class NotificationService:
    """Service for sending workflow notifications via Discord."""

    def __init__(self):
        """Initialize notification service with configuration from environment variables."""
        self.discord_webhook_url = os.getenv("DISCORD_WEBHOOK_URL")

        # Check if Discord is configured
        self.enabled = bool(self.discord_webhook_url)

    def send_notification(
        self,
        title: str,
        message: str,
        status: str = "info",
        details: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Send notification via Discord webhook.

        Args:
            title: Notification title
            message: Notification message
            status: Status type (success, error, warning, info)
            details: Additional details dictionary

        Returns:
            True if notification sent successfully, False otherwise
        """
        if not self.enabled:
            logger = get_run_logger()
            logger.warning("Discord webhook not configured. Skipping notification.")
            return False

        try:
            return self._send_discord(title, message, status, details)
        except Exception as e:
            logger = get_run_logger()
            logger.error(f"Failed to send Discord notification: {e}")
            return False

    def _send_discord(
        self,
        title: str,
        message: str,
        status: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Send Discord notification via webhook."""
        # Discord color mapping
        color_map = {
            "success": 0x00FF00,  # Green
            "error": 0xFF0000,  # Red
            "warning": 0xFFFF00,  # Yellow
            "info": 0x0099FF,  # Blue
        }

        embed = {
            "title": title,
            "description": message,
            "color": color_map.get(status, 0x0099FF),
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": "ML Training Pipeline"},
        }

        if details:
            fields = []
            for key, value in details.items():
                fields.append(
                    {
                        "name": key.replace("_", " ").title(),
                        "value": str(value)[:1024],  # Discord field value limit
                        "inline": True,
                    }
                )
            embed["fields"] = fields

        payload = {"embeds": [embed]}

        try:
            response = httpx.post(self.discord_webhook_url, json=payload, timeout=10.0)
            response.raise_for_status()
            return True
        except Exception as e:
            logger = get_run_logger()
            logger.error(f"Discord notification failed: {e}")
            return False

    def notify_success(
        self, flow_name: str, duration: float, details: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Send success notification."""
        title = f"✅ {flow_name} Completed Successfully"
        message = f"The ML training pipeline completed successfully in {duration:.2f} seconds."

        if details is None:
            details = {}
        details["duration_seconds"] = f"{duration:.2f}s"

        return self.send_notification(title, message, "success", details)

    def notify_failure(
        self, flow_name: str, error: str, details: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Send failure notification."""
        title = f"❌ {flow_name} Failed"
        message = f"The ML training pipeline failed with error: {error}"

        if details is None:
            details = {}
        details["error"] = str(error)[:500]  # Truncate long errors

        return self.send_notification(title, message, "error", details)

    def notify_warning(
        self, flow_name: str, warning: str, details: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Send warning notification."""
        title = f"⚠️ {flow_name} Warning"
        message = warning

        return self.send_notification(title, message, "warning", details)
