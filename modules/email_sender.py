"""
Send approved replies by SMTP. Configure with environment variables:
  SMTP_HOST, SMTP_PORT (default 587; 465 = SSL), SMTP_USER, SMTP_PASSWORD, SMTP_FROM,
  SMTP_STARTTLS (default "true", ignored on port 465)
Works with Gmail/Outlook (use an app password), SendGrid, Mailgun, or any SMTP relay.
"""
import os
import re
import smtplib
import ssl
from email.message import EmailMessage

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def smtp_configured() -> bool:
    return bool(os.getenv("SMTP_HOST") and os.getenv("SMTP_FROM"))


def is_valid_address(address: str) -> bool:
    return bool(_EMAIL_RE.match(address or ""))


def split_subject(draft: str, ticket_id: str) -> tuple[str, str]:
    """The LLM draft starts with 'SUBJECT: ...'; split it from the body."""
    lines = draft.strip().splitlines()
    if lines and lines[0].upper().startswith("SUBJECT:"):
        return lines[0].split(":", 1)[1].strip(), "\n".join(lines[1:]).strip()
    return f"Re: your support request - Ref: {ticket_id}", draft.strip()


def send_email(to: str, subject: str, body: str) -> None:
    """Raises ValueError for a bad address, smtplib/OSError errors if sending fails."""
    if not is_valid_address(to):
        raise ValueError(f"Invalid email address: {to!r}")

    msg = EmailMessage()
    msg["From"] = os.environ["SMTP_FROM"]
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)

    host, port = os.environ["SMTP_HOST"], int(os.getenv("SMTP_PORT", "587"))
    context = ssl.create_default_context()
    if port == 465:
        server = smtplib.SMTP_SSL(host, port, context=context, timeout=30)
    else:
        server = smtplib.SMTP(host, port, timeout=30)
    with server:
        if port != 465 and os.getenv("SMTP_STARTTLS", "true").lower() != "false":
            server.starttls(context=context)
        if os.getenv("SMTP_USER"):
            server.login(os.environ["SMTP_USER"], os.getenv("SMTP_PASSWORD", ""))
        server.send_message(msg)
