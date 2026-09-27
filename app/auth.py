"""Minimal first-party authentication for the local enterprise demo."""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import time
from uuid import uuid4

from sqlalchemy import text

from .config import get_settings
from .secure_demo import Actor, actor_from_email, engine

TOKEN_TTL_SECONDS = 60 * 60 * 8
DEMO_PASSWORD = "Demo!2026"


def _password_hash(password: str, salt: bytes | None = None) -> str:
    salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 310_000)
    return f"{salt.hex()}${digest.hex()}"


def _verify_password(password: str, encoded: str) -> bool:
    salt_hex, expected = encoded.split("$", 1)
    actual = _password_hash(password, bytes.fromhex(salt_hex)).split("$", 1)[1]
    return hmac.compare_digest(actual, expected)


def _secret() -> bytes:
    # A stable secret must be configured outside local teaching mode.
    return os.getenv("AUTH_SECRET", "local-demo-secret-change-me").encode()


def _token(email: str) -> str:
    payload = f"{email}|{int(time.time()) + TOKEN_TTL_SECONDS}".encode()
    signature = hmac.new(_secret(), payload, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(payload + b"." + signature).decode().rstrip("=")


def _decode_token(token: str) -> str:
    raw = base64.urlsafe_b64decode(token + "=" * (-len(token) % 4))
    payload, signature = raw.rsplit(b".", 1)
    if not hmac.compare_digest(signature, hmac.new(_secret(), payload, hashlib.sha256).digest()):
        raise ValueError("Invalid session token")
    email, expires_at = payload.decode().rsplit("|", 1)
    if int(expires_at) < time.time():
        raise ValueError("Session expired")
    return email


def initialize_auth() -> None:
    """Idempotent migration plus local accounts for the seeded teaching users."""
    with engine().begin() as connection:
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS auth_accounts (
                user_id UUID PRIMARY KEY REFERENCES app_users(user_id) ON DELETE CASCADE,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """))
        users = connection.execute(text("SELECT user_id FROM app_users")).mappings().all()
        password_hash = _password_hash(DEMO_PASSWORD)
        for row in users:
            connection.execute(text("""
                INSERT INTO auth_accounts (user_id, password_hash) VALUES (:user_id, :password_hash)
                ON CONFLICT (user_id) DO NOTHING
            """), {"user_id": row["user_id"], "password_hash": password_hash})


def login(email: str, password: str) -> tuple[str, Actor]:
    with engine().connect() as connection:
        row = connection.execute(text("""
            SELECT a.password_hash FROM auth_accounts a JOIN app_users u ON u.user_id = a.user_id
            WHERE u.email = :email AND u.status = 'active'
        """), {"email": email.lower()}).mappings().first()
    if not row or not _verify_password(password, row["password_hash"]):
        raise ValueError("Email or password is incorrect")
    actor = actor_from_email(email.lower())
    return _token(actor.email), actor


def actor_from_bearer(authorization: str | None) -> Actor:
    if not authorization or not authorization.startswith("Bearer "):
        raise ValueError("Sign in is required")
    return actor_from_email(_decode_token(authorization[7:]))


def register_company(company_name: str, email: str, display_name: str, password: str) -> tuple[str, Actor]:
    email = email.lower()
    tenant_id, user_id = str(uuid4()), str(uuid4())
    with engine().begin() as connection:
        exists = connection.execute(text("SELECT 1 FROM app_users WHERE email=:email"), {"email": email}).first()
        if exists:
            raise ValueError("This email is already registered")
        connection.execute(text("""INSERT INTO tenants (tenant_id, tenant_name, industry, plan, created_at)
            VALUES (:tenant_id, :tenant_name, 'Business services', 'trial', NOW())"""), {"tenant_id": tenant_id, "tenant_name": company_name})
        connection.execute(text("""INSERT INTO app_users (user_id, tenant_id, email, display_name, status, created_at)
            VALUES (:user_id, :tenant_id, :email, :display_name, 'active', NOW())"""), {"user_id": user_id, "tenant_id": tenant_id, "email": email, "display_name": display_name})
        connection.execute(text("""INSERT INTO user_role_assignments (user_id, role_id)
            SELECT :user_id, role_id FROM roles WHERE role_name='tenant_admin'"""), {"user_id": user_id})
        connection.execute(text("INSERT INTO auth_accounts (user_id, password_hash) VALUES (:user_id, :password_hash)"), {"user_id": user_id, "password_hash": _password_hash(password)})
    actor = actor_from_email(email)
    return _token(email), actor


def invite_user(actor: Actor, email: str, display_name: str, role: str, password: str) -> None:
    if actor.role not in {"tenant_admin", "security_admin"}:
        raise PermissionError("Only a tenant administrator can invite users")
    email = email.lower(); user_id = str(uuid4())
    with engine().begin() as connection:
        if connection.execute(text("SELECT 1 FROM app_users WHERE email=:email"), {"email": email}).first():
            raise ValueError("This email is already registered")
        role_id = connection.execute(text("SELECT role_id FROM roles WHERE role_name=:role"), {"role": role}).scalar()
        if not role_id:
            raise ValueError("Unknown role")
        connection.execute(text("""INSERT INTO app_users (user_id, tenant_id, email, display_name, status, created_at)
            VALUES (:user_id, :tenant_id, :email, :display_name, 'active', NOW())"""), {"user_id": user_id, "tenant_id": actor.tenant_id, "email": email, "display_name": display_name})
        connection.execute(text("INSERT INTO user_role_assignments (user_id, role_id) VALUES (:user_id, :role_id)"), {"user_id": user_id, "role_id": role_id})
        connection.execute(text("INSERT INTO auth_accounts (user_id, password_hash) VALUES (:user_id, :password_hash)"), {"user_id": user_id, "password_hash": _password_hash(password)})


def tenant_users(actor: Actor) -> list[dict]:
    if actor.role not in {"tenant_admin", "security_admin"}:
        raise PermissionError("Administrator access is required")
    with engine().connect() as connection:
        return [dict(row) for row in connection.execute(text("""
            SELECT u.email, u.display_name, u.status, r.role_name
            FROM app_users u JOIN user_role_assignments ura ON ura.user_id=u.user_id
            JOIN roles r ON r.role_id=ura.role_id WHERE u.tenant_id=:tenant_id ORDER BY u.created_at DESC
        """), {"tenant_id": actor.tenant_id}).mappings()]
