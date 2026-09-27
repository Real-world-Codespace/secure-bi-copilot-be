"""Tenant-scoped, prompt-safe demo service used by the enterprise lab."""
from __future__ import annotations

import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from uuid import uuid4

from sqlalchemy import create_engine, text
from redis import Redis

from .config import get_settings

INJECTION_PATTERNS = re.compile(
    r"ignore\s+(all|previous)|system\s*(prompt|override)|developer\s+message|"
    r"reveal\s+(secret|prompt|api[_ ]?key)|export\s+(all|the).*(payroll|email)|"
    r"bypass\s+(policy|guard)|jailbreak|"
    r"bỏ\s+qua.*(hướng\s+dẫn|quy\s+tắc|chính\s+sách)|"
    r"tiết\s+lộ.*(prompt|bí\s+mật|api[_ ]?key)|"
    r"xuất.*(toàn\s+bộ|tất\s+cả).*(lương|payroll|email)",
    re.IGNORECASE,
)
SENSITIVE_ROLES = {"security_admin", "tenant_admin"}


@dataclass
class Actor:
    user_id: str
    tenant_id: str
    email: str
    role: str


class LocalRateLimiter:
    """Safe development fallback. Production deployments use Redis at the edge."""
    def __init__(self) -> None:
        self._events: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str, limit: int) -> tuple[bool, int]:
        now = time.monotonic(); window = 60.0; events = self._events[key]
        while events and events[0] <= now - window: events.popleft()
        if len(events) >= limit:
            return False, max(1, int(window - (now - events[0])))
        events.append(now)
        return True, 0


limiter = LocalRateLimiter()
_redis_client: Redis | None = None


def _redis() -> Redis | None:
    global _redis_client
    url = get_settings().redis_url
    if not url:
        return None
    if _redis_client is None:
        _redis_client = Redis.from_url(url, decode_responses=True, socket_connect_timeout=0.25)
    return _redis_client


def check_rate_limit(key: str) -> tuple[bool, int]:
    """Distributed counter in Redis, with an explicit local dev fallback."""
    limit = get_settings().rate_limit_per_minute
    try:
        client = _redis()
        if client is not None:
            redis_key = f"secure-bi:rate:{key}"
            count = client.incr(redis_key)
            if count == 1:
                client.expire(redis_key, 60)
            ttl = max(1, client.ttl(redis_key))
            return count <= limit, ttl
    except Exception:
        # The fallback keeps local teaching mode usable; production should alert on Redis failure.
        pass
    return limiter.allow(key, limit)


def engine():
    return create_engine(get_settings().database_url, pool_pre_ping=True)


def actor_from_email(email: str) -> Actor:
    sql = text("""
        SELECT u.user_id::text, u.tenant_id::text, u.email, r.role_name AS role
        FROM app_users u
        JOIN user_role_assignments ura ON ura.user_id = u.user_id
        JOIN roles r ON r.role_id = ura.role_id
        WHERE u.email = :email AND u.status = 'active'
        LIMIT 1
    """)
    with engine().connect() as connection:
        row = connection.execute(sql, {"email": email}).mappings().first()
    if not row:
        raise ValueError("Unknown demo user")
    return Actor(**dict(row))


def list_demo_users() -> list[dict]:
    sql = text("""
        SELECT u.email, u.display_name, t.tenant_name, r.role_name
        FROM app_users u JOIN tenants t ON t.tenant_id = u.tenant_id
        JOIN user_role_assignments ura ON ura.user_id = u.user_id
        JOIN roles r ON r.role_id = ura.role_id ORDER BY t.tenant_name, r.role_name
    """)
    with engine().connect() as connection:
        return [dict(row) for row in connection.execute(sql).mappings()]


def assess_prompt(prompt: str) -> tuple[bool, str]:
    if INJECTION_PATTERNS.search(prompt):
        return False, "Prompt was blocked because it attempts to override policy or request protected data."
    if len(prompt) > 1500:
        return False, "Prompt exceeds the maximum allowed length."
    return True, "allowed"


def audit(actor: Actor, event_type: str, decision: str, metadata: str = '{"source":"api"}') -> None:
    stmt = text("""INSERT INTO copilot_audit_events
        (event_id, tenant_id, user_id, created_at, event_type, decision, trace_id, metadata)
        VALUES (:event_id, :tenant_id, :user_id, NOW(), :event_type, :decision, :trace_id, CAST(:metadata AS jsonb))""")
    with engine().begin() as connection:
        connection.execute(stmt, {"event_id": str(uuid4()), "tenant_id": actor.tenant_id, "user_id": actor.user_id,
            "event_type": event_type, "decision": decision, "trace_id": f"tr_{uuid4().hex[:12]}", "metadata": metadata})


def _rows(sql: str, actor: Actor) -> list[dict]:
    # tenant_id is bound server-side; it never comes from model or browser input.
    with engine().connect() as connection:
        return [dict(row) for row in connection.execute(text(sql), {"tenant_id": actor.tenant_id}).mappings()]


def widget(title: str, kind: str, sql: str, actor: Actor, **fields) -> dict:
    return {"id": uuid4().hex, "title": title, "description": fields.pop("description", ""), "chart_type": kind,
        "data": _rows(sql, actor), "sql": sql.replace(":tenant_id", "[server-bound tenant]"), "layout": fields.pop("layout", {"col_span": 6, "row_span": 1}), **fields}


def dashboard(actor: Actor, prompt: str) -> dict:
    lower = prompt.lower()
    if any(word in lower for word in ("lương", "payroll", "salary", "nhân viên")) and actor.role not in {"hr_manager", *SENSITIVE_ROLES}:
        audit(actor, "authorization_denied", "blocked")
        raise PermissionError("Your role cannot access HR or payroll data.")
    if any(word in lower for word in ("tồn", "inventory", "kho", "stock")):
        widgets = [widget("SKU below reorder point", "kpi", "SELECT COUNT(*) AS value FROM v_inventory_risk WHERE tenant_id=:tenant_id AND at_risk", actor, value_field="value", number_format="integer", layout={"col_span": 3, "row_span": 1}), widget("Inventory risk by region", "bar", "SELECT region, COUNT(*) AS at_risk_skus FROM v_inventory_risk WHERE tenant_id=:tenant_id AND at_risk GROUP BY region ORDER BY at_risk_skus DESC", actor, x_field="region", y_fields=["at_risk_skus"], number_format="integer")]
        title, message = "Inventory risk dashboard", "Results are scoped to the authenticated tenant."
    elif any(word in lower for word in ("ticket", "support", "chăm sóc")):
        widgets = [widget("Open tickets", "kpi", "SELECT COUNT(*) AS value FROM support_tickets WHERE tenant_id=:tenant_id AND status IN ('open','pending')", actor, value_field="value", number_format="integer", layout={"col_span": 3, "row_span": 1}), widget("Resolution time by category", "bar", "SELECT category, ROUND(AVG(resolution_hours), 1) AS avg_hours FROM support_tickets WHERE tenant_id=:tenant_id GROUP BY category ORDER BY avg_hours DESC", actor, x_field="category", y_fields=["avg_hours"], number_format="decimal")]
        title, message = "Support operations dashboard", "Ticket text is treated as untrusted reference data."
    else:
        widgets = [widget("Revenue", "kpi", "SELECT ROUND(SUM(total_amount),2) AS value FROM orders WHERE tenant_id=:tenant_id AND status NOT IN ('cancelled','refunded')", actor, value_field="value", number_format="currency", layout={"col_span": 3, "row_span": 1}), widget("Orders", "kpi", "SELECT COUNT(*) AS value FROM orders WHERE tenant_id=:tenant_id AND status NOT IN ('cancelled','refunded')", actor, value_field="value", number_format="integer", layout={"col_span": 3, "row_span": 1}), widget("Monthly revenue", "line", "SELECT TO_CHAR(order_date, 'YYYY-MM') AS month, ROUND(SUM(total_amount),2) AS revenue FROM orders WHERE tenant_id=:tenant_id AND status NOT IN ('cancelled','refunded') GROUP BY month ORDER BY month", actor, x_field="month", y_fields=["revenue"], number_format="currency"), widget("Top categories", "bar", "SELECT p.category, ROUND(SUM(oi.line_total),2) AS revenue FROM order_items oi JOIN orders o ON o.order_id=oi.order_id JOIN products p ON p.product_id=oi.product_id WHERE o.tenant_id=:tenant_id AND o.status NOT IN ('cancelled','refunded') GROUP BY p.category ORDER BY revenue DESC", actor, x_field="category", y_fields=["revenue"], number_format="currency")]
        title, message = "Tenant sales dashboard", "SQL is selected from server-owned templates and tenant-scoped before execution."
    audit(actor, "copilot_query", "allowed")
    return {"title": title, "message": message, "mode": "heuristic", "widgets": widgets, "security": {"tenant": actor.email.split('@')[0], "role": actor.role, "prompt_decision": "allowed"}}
