"""Email alert sender via SMTP.

Sends plaintext or HTML email via Python's smtplib.

Usage:
    from pipeline.monitor.email_alert import send_email_alert, EmailConfig
    cfg = EmailConfig(
        smtp_host="smtp.example.com",
        smtp_port=587,
        username="alert@example.com",
        password="secret",
        from_addr="LLMTagger <alert@example.com>",
        use_tls=True,
    )
    send_email_alert(cfg, to="ops@example.com", subject="Token 告警", body="...")

For MVP the EmailConfig can be left as None — send_email_alert() will print
a formatted preview and return False without actually sending.
"""
from __future__ import annotations

import smtplib
from dataclasses import dataclass
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional


@dataclass
class EmailConfig:
    smtp_host: str = ""
    smtp_port: int = 587
    username: str = ""
    password: str = ""
    from_addr: str = ""
    use_tls: bool = True


def send_email_alert(
    cfg: Optional[EmailConfig],
    to: str,
    subject: str,
    body: str,
    html_body: Optional[str] = None,
) -> bool:
    """Send an email alert.

    Returns True on success. If cfg is None or smtp_host is empty, prints a
    preview to stdout and returns False (MVP stub mode).
    """
    if cfg is None or not cfg.smtp_host:
        _print_preview(to, subject, body)
        return False

    try:
        msg = _build_message(cfg.from_addr or cfg.username, to, subject, body, html_body)
        if cfg.use_tls:
            server = smtplib.SMTP(cfg.smtp_host, cfg.smtp_port, timeout=15)
            server.ehlo()
            server.starttls()
        else:
            server = smtplib.SMTP_SSL(cfg.smtp_host, cfg.smtp_port, timeout=15)
        if cfg.username:
            server.login(cfg.username, cfg.password)
        server.sendmail(cfg.from_addr or cfg.username, [to], msg.as_string())
        server.quit()
        return True
    except Exception as e:
        import sys
        print(f"[EmailAlert] send failed: {e}", file=sys.stderr)
        return False


def build_html_body(
    title: str,
    body: str,
    fields: Optional[dict] = None,
) -> str:
    """Build a minimal HTML email body."""
    rows = ""
    if fields:
        rows = "".join(
            f"<tr><td style='padding:4px 8px;font-weight:bold'>{k}</td>"
            f"<td style='padding:4px 8px'>{v}</td></tr>"
            for k, v in fields.items()
        )
        rows = f"<table border='0' cellpadding='0' cellspacing='0'>{rows}</table>"
    return f"""
    <html><body style="font-family:sans-serif;color:#333">
    <h3 style="color:#c0392b">{title}</h3>
    <p>{body}</p>
    {rows}
    <hr/><small>LLMTagger 自动告警</small>
    </body></html>
    """


def _build_message(
    from_addr: str,
    to: str,
    subject: str,
    body: str,
    html_body: Optional[str],
) -> MIMEMultipart:
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to
    msg.attach(MIMEText(body, "plain", "utf-8"))
    if html_body:
        msg.attach(MIMEText(html_body, "html", "utf-8"))
    return msg


def _print_preview(to: str, subject: str, body: str) -> None:
    print(f"[EmailAlert Stub] To={to}  Subject={subject}")
    print(f"  {body[:200]}")
