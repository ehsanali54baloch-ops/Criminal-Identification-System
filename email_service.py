#!/usr/bin/env python3
"""
==============================================================================
AI-Based Face Detection & Criminal Identification System
Admin Email Alerts (email_service.py)
Team: Ehsan Ali (Leader), Asif Ali, Sabir Ali
==============================================================================
Sends the admin an email with the suspect's details and the webcam snapshot
attached, using normal SMTP (works with Gmail, Outlook, etc.).

Settings come from environment variables or a .env file (never commit it!):

    SMTP_HOST=smtp.gmail.com
    SMTP_PORT=587
    SMTP_USER=yourproject@gmail.com
    SMTP_PASSWORD=xxxx xxxx xxxx xxxx      <- Gmail "App Password", not your normal password
    ADMIN_EMAIL=admin@example.com

Emails are sent in a background thread so the camera never freezes.
"""

import os
import smtplib
import sqlite3
import threading
from datetime import datetime
from email.message import EmailMessage
from typing import Dict, Any, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DB_PATH = "criminals.db"


def get_settings() -> Dict[str, Any]:
    return {
        "host": os.environ.get("SMTP_HOST", "smtp.gmail.com"),
        "port": int(os.environ.get("SMTP_PORT", "587")),
        "user": os.environ.get("SMTP_USER", "").strip(),
        "password": os.environ.get("SMTP_PASSWORD", "").replace(" ", ""),
        "admin": os.environ.get("ADMIN_EMAIL", "").strip(),
    }


def is_configured() -> bool:
    s = get_settings()
    return bool(s["user"] and s["password"] and s["admin"])


def configure(admin_email: str, smtp_user: str, smtp_password: str,
              smtp_host: str = "smtp.gmail.com", smtp_port: int = 587):
    """Set email settings at runtime (from the dashboard's Email Settings tab)."""
    if admin_email:
        os.environ["ADMIN_EMAIL"] = admin_email.strip()
    if smtp_user:
        os.environ["SMTP_USER"] = smtp_user.strip()
    if smtp_password:
        os.environ["SMTP_PASSWORD"] = smtp_password.strip()
    os.environ["SMTP_HOST"] = (smtp_host or "smtp.gmail.com").strip()
    os.environ["SMTP_PORT"] = str(int(smtp_port or 587))


def _send(msg: EmailMessage) -> Tuple[bool, str]:
    s = get_settings()
    if not is_configured():
        return False, "Email not configured (set ADMIN_EMAIL, SMTP_USER, SMTP_PASSWORD)."
    msg["From"] = s["user"]
    msg["To"] = s["admin"]
    try:
        if s["port"] == 465:
            server = smtplib.SMTP_SSL(s["host"], s["port"], timeout=20)
        else:
            server = smtplib.SMTP(s["host"], s["port"], timeout=20)
            server.ehlo()
            if server.has_extn("starttls"):
                server.starttls()
                server.ehlo()
        if server.has_extn("auth"):
            server.login(s["user"], s["password"])
        server.send_message(msg)
        server.quit()
        return True, f"Sent to {s['admin']}"
    except smtplib.SMTPAuthenticationError:
        return False, ("Login failed. For Gmail you must use a 16-letter App Password "
                       "(Google Account > Security > 2-Step Verification > App passwords).")
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def build_alert_email(suspect: Dict[str, Any], mask_info: Dict[str, Any],
                      snapshot_path: str) -> EmailMessage:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = EmailMessage()
    msg["Subject"] = (f"🚨 ALERT: {suspect['name']} ({suspect['suspect_id']}) detected "
                      f"- {suspect['confidence_score']}% match")
    text = (
        f"SECURITY ALERT - Criminal identified on webcam\n\n"
        f"Name:          {suspect['name']}\n"
        f"Suspect ID:    {suspect['suspect_id']}\n"
        f"Crime:         {suspect.get('crime_category', '')}\n"
        f"Threat level:  {suspect.get('threat_level', '')}\n"
        f"Status:        {suspect.get('status', '')}\n"
        f"Face match:    {suspect['confidence_score']}%\n"
        f"Mask:          {mask_info.get('status', '')}\n"
        f"Camera:        Webcam Surveillance Stream 1\n"
        f"Time:          {now}\n\n"
        f"The webcam snapshot is attached.\n"
    )
    msg.set_content(text)
    msg.add_alternative(f"""
    <div style="font-family:Arial,sans-serif;max-width:560px">
      <div style="background:#b91c1c;color:#fff;padding:14px 18px;border-radius:8px 8px 0 0;font-size:18px;font-weight:bold">
        🚨 Criminal Identified on Webcam
      </div>
      <table style="border:1px solid #ddd;border-top:none;width:100%;padding:12px;font-size:14px">
        <tr><td><b>Name</b></td><td>{suspect['name']}</td></tr>
        <tr><td><b>Suspect ID</b></td><td>{suspect['suspect_id']}</td></tr>
        <tr><td><b>Crime</b></td><td>{suspect.get('crime_category', '')}</td></tr>
        <tr><td><b>Threat level</b></td><td style="color:#b91c1c"><b>{suspect.get('threat_level', '')}</b></td></tr>
        <tr><td><b>Face match</b></td><td>{suspect['confidence_score']}%</td></tr>
        <tr><td><b>Mask</b></td><td>{mask_info.get('status', '')}</td></tr>
        <tr><td><b>Time</b></td><td>{now}</td></tr>
      </table>
      <p style="font-size:13px;color:#555">Webcam snapshot attached.</p>
    </div>""", subtype="html")
    if snapshot_path and os.path.exists(snapshot_path):
        with open(snapshot_path, "rb") as f:
            msg.add_attachment(f.read(), maintype="image", subtype="jpeg",
                               filename=os.path.basename(snapshot_path))
    return msg


def _set_email_status(alert_id: int, status: str):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("UPDATE incident_alerts SET email_status = ? WHERE id = ?", (status, alert_id))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[Email Status DB Error]: {e}")


def send_alert_async(alert_id: int, suspect: Dict[str, Any], mask_info: Dict[str, Any],
                     snapshot_path: str):
    """Send the alert email in the background and record the result in the log."""
    def worker():
        ok, info = _send(build_alert_email(suspect, mask_info, snapshot_path))
        print(f"[EMAIL] {'✅' if ok else '❌'} {info}")
        _set_email_status(alert_id, ("SENT" if ok else "FAILED: ") + ("" if ok else info[:120]))
    if not is_configured():
        _set_email_status(alert_id, "NOT CONFIGURED")
        return
    _set_email_status(alert_id, "SENDING...")
    threading.Thread(target=worker, daemon=True).start()


def send_test_email() -> Tuple[bool, str]:
    msg = EmailMessage()
    msg["Subject"] = "✅ Test email - Face Recognition Surveillance System"
    msg.set_content("Email alerts are working. You will receive a message with a "
                    "snapshot whenever a criminal is identified on the webcam.")
    return _send(msg)
