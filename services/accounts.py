"""Local email/password accounts for the Streamlit application."""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import uuid
from datetime import datetime, timezone

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError


PBKDF2_ITERATIONS = 600_000
PBKDF2_SCHEME = "pbkdf2_sha256"
EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@.]+(?:\.[^\s@.]+)+$")
MAX_PASSWORD_LENGTH = 256


class DuplicateEmailError(ValueError):
    """Raised when an account already uses the normalized email address."""


def normalize_email(email: str) -> str:
    return str(email or "").strip().casefold()


def is_valid_email(email: str) -> bool:
    normalized = normalize_email(email)
    return len(normalized) <= 254 and bool(EMAIL_PATTERN.fullmatch(normalized))


def hash_password(password: str) -> str:
    if not isinstance(password, str) or not password or len(password) > MAX_PASSWORD_LENGTH:
        raise ValueError("Password length is invalid.")
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS
    )
    return f"{PBKDF2_SCHEME}${PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, encoded_hash: str) -> bool:
    if not isinstance(password, str) or not isinstance(encoded_hash, str):
        return False
    try:
        scheme, iteration_text, salt_hex, digest_hex = encoded_hash.split("$", 3)
        iterations = int(iteration_text)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(digest_hex)
        if (
            scheme != PBKDF2_SCHEME
            or iterations < PBKDF2_ITERATIONS
            or iterations > 2_000_000
            or len(salt) != 16
            or len(expected) != 32
            or len(password) > MAX_PASSWORD_LENGTH
        ):
            return False
    except (TypeError, ValueError):
        return False

    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(actual, expected)


def register_account(engine, full_name: str, email: str, password: str) -> dict[str, str]:
    name = str(full_name or "").strip()
    normalized_email = normalize_email(email)
    if not name or len(name) > 120:
        raise ValueError("Enter a name no longer than 120 characters.")
    if not is_valid_email(normalized_email):
        raise ValueError("Enter a valid email address.")
    if not isinstance(password, str) or len(password) < 12 or len(password) > MAX_PASSWORD_LENGTH:
        raise ValueError("Use a password between 12 and 256 characters.")

    account = {
        "user_id": uuid.uuid4().hex,
        "full_name": name,
        "email": normalized_email,
        "password_hash": hash_password(password),
    }
    try:
        with engine.begin() as connection:
            connection.execute(
                text("""
                    INSERT INTO app_users (user_id, full_name, email, password_hash, created_at)
                    VALUES (:user_id, :full_name, :email, :password_hash, :created_at)
                """),
                {**account, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")},
            )
    except IntegrityError as exc:
        raise DuplicateEmailError("An account already exists for this email address.") from exc

    return {key: account[key] for key in ("user_id", "full_name", "email")}


def authenticate_account(engine, email: str, password: str) -> dict[str, str] | None:
    normalized_email = normalize_email(email)
    if not is_valid_email(normalized_email) or not password or len(password) > MAX_PASSWORD_LENGTH:
        return None
    with engine.connect() as connection:
        row = connection.execute(
            text("SELECT user_id, full_name, email, password_hash FROM app_users WHERE email=:email"),
            {"email": normalized_email},
        ).mappings().first()
    if not row:
        hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes(16), PBKDF2_ITERATIONS)
        return None
    if not verify_password(password, row["password_hash"]):
        return None
    return {key: row[key] for key in ("user_id", "full_name", "email")}
