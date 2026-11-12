"""LangGraph Data Augmented Generation workflow for the secure executive copilot."""
from __future__ import annotations

import json
from functools import lru_cache
from decimal import Decimal
from typing import Any, Literal
from uuid import uuid4

from langchain_openai import ChatOpenAI
from langchain_core.rate_limiters import InMemoryRateLimiter
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlglot import exp, parse_one
from typing_extensions import TypedDict

from .config import get_settings
from .secure_demo import Actor, LLMRateLimitExceeded, _check_rate_limit, audit, engine


CATALOG = {
    "sales_manager": {"orders", "order_items", "products", "customers", "payments", "shipments", "returns", "v_sales_daily"},
    "operations_manager": {"inventory_snapshots", "warehouses", "products", "support_tickets", "v_inventory_risk"},
    "hr_manager": {"employees", "payroll"},
    "security_admin": {"orders", "order_items", "products", "customers", "payments", "shipments", "returns", "inventory_snapshots", "warehouses", "support_tickets", "v_sales_daily", "v_inventory_risk"},
}
CATALOG["tenant_admin"] = set().union(*CATALOG.values())

TABLE_GUIDE = {
    "orders": "orders(order_id, tenant_id, customer_id, warehouse_id, order_date, channel, status, subtotal, discount_amount, shipping_fee, total_amount)",
    "order_items": "order_items(order_item_id, order_id, product_id, quantity, unit_price, unit_cost, line_discount, line_total)",
    "products": "products(product_id, tenant_id, sku, product_name, category, unit_cost, list_price, active)",
    "customers": "customers(customer_id, tenant_id, customer_name, segment, region, email, phone, created_at)",
    "payments": "payments(payment_id, order_id, paid_at, method, status, amount)",
    "shipments": "shipments(shipment_id, order_id, carrier, shipped_at, delivered_at, promised_date, status)",
    "returns": "returns(return_id, order_id, returned_at, reason, refund_amount, status)",
    "v_sales_daily": "v_sales_daily(tenant_id, day, channel, order_count, revenue)",
    "inventory_snapshots": "inventory_snapshots(snapshot_id, tenant_id, warehouse_id, product_id, snapshot_date, on_hand_units, reserved_units, reorder_point)",
    "warehouses": "warehouses(warehouse_id, tenant_id, warehouse_name, region, capacity_units)",
    "support_tickets": "support_tickets(ticket_id, tenant_id, customer_id, order_id, created_at, channel, priority, category, status, resolution_hours)",
    "v_inventory_risk": "v_inventory_risk(tenant_id, snapshot_date, region, category, sku, on_hand_units, reserved_units, reorder_point, at_risk)",
    "employees": "employees(employee_id, tenant_id, employee_name, department, job_title, region, hired_at, employment_status)",
    "payroll": "payroll(payroll_id, employee_id, pay_period, base_salary, bonus)",
}


class Relevance(BaseModel):
    relevance: Literal["relevant", "not_relevant"]


class RelevanceAndSQL(BaseModel):
    """The first LLM node combines two dependent planning decisions.

    Keeping them together avoids a separate network round trip while the DAG
    still exposes check_relevance -> generate_sql as distinct control nodes.
    """
    relevance: Literal["relevant", "not_relevant"]
    sql_query: str | None = Field(default=None, max_length=6000)


class SQLDraft(BaseModel):
    sql_query: str = Field(min_length=12, max_length=6000)


class Answer(BaseModel):
    title: str = Field(min_length=4, max_length=90)
    message: str = Field(min_length=10, max_length=700)


MAX_SQL_REPAIR_ATTEMPTS = 2


class DAGState(TypedDict, total=False):
    question: str
    sql_query: str
    query_rows: list[dict[str, Any]]
    attempts: int
    relevance: str
    sql_error: str
    failure_message: str
    answer: Answer


def _catalog_text(actor: Actor) -> str:
    tables = sorted(CATALOG.get(actor.role, set()))
    joins = (
        "Join keys: orders.order_id=order_items.order_id; order_items.product_id=products.product_id; "
        "orders.customer_id=customers.customer_id; orders.warehouse_id=warehouses.warehouse_id; "
        "payroll.employee_id=employees.employee_id."
    )
    return "\n".join([*(f"- {TABLE_GUIDE[name]}" for name in tables), joins])


@lru_cache(maxsize=8)
def _worker_rate_limiter(requests_per_second: float, max_burst: int) -> InMemoryRateLimiter:
    """One process-local token bucket shared by every DAG node in this worker.

    It smooths outbound traffic before requests reach OpenAI. It deliberately
    does not replace the Redis quota because it cannot coordinate containers or
    identify a tenant/user.
    """
    return InMemoryRateLimiter(
        requests_per_second=requests_per_second,
        max_bucket_size=max_burst,
    )


def _llm(actor: Actor) -> ChatOpenAI:
    settings = get_settings()
    allowed, retry_after = _check_rate_limit("llm", f"{actor.tenant_id}:{actor.user_id}:{settings.openai_model}", settings.llm_rate_limit_per_minute)
    if not allowed:
        audit(actor, "llm_rate_limit_exceeded", "blocked", json.dumps({"model": settings.openai_model, "retry_after_seconds": retry_after}))
        raise LLMRateLimitExceeded(retry_after)
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0,
        rate_limiter=_worker_rate_limiter(
            settings.llm_worker_requests_per_second,
            settings.llm_worker_max_burst,
        ),
        max_retries=settings.openai_max_retries,
    )


def _validate_sql(sql: str, actor: Actor) -> str:
    candidate = sql.strip().rstrip(";")
    if not candidate or ";" in candidate or ":tenant_id" not in candidate:
        raise ValueError("SQL must be one read-only statement with the server-bound :tenant_id filter")
    expression = parse_one(candidate, read="postgres")
    if not isinstance(expression, (exp.Select, exp.Union)):
        raise ValueError("Only SELECT queries are allowed")
    tables = {table.name.lower() for table in expression.find_all(exp.Table)}
    unknown = tables - CATALOG.get(actor.role, set())
    if not tables or unknown:
        raise ValueError(f"Unauthorized table(s): {', '.join(sorted(unknown))}")
    banned = (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Command)
    if any(expression.find(kind) for kind in banned):
        raise ValueError("Only read-only SQL is allowed")
    # sqlglot serializes named placeholders as ``%(tenant_id)s`` for its
    # PostgreSQL dialect. SQLAlchemy's ``text()`` expects ``:tenant_id`` and
    # otherwise escapes the percent sign before handing it to psycopg.
    normalized = expression.limit(get_settings().max_query_rows).sql(dialect="postgres")
    return normalized.replace("%(tenant_id)s", ":tenant_id")


def _execute(sql: str, actor: Actor) -> list[dict[str, Any]]:
    with engine().connect() as connection:
        rows = connection.execute(text(sql), {"tenant_id": actor.tenant_id}).mappings().all()
    return [{key: float(value) if isinstance(value, Decimal) else value for key, value in row.items()} for row in rows]


def _widget(rows: list[dict[str, Any]], sql: str) -> dict:
    if len(rows) == 1 and len(rows[0]) == 1:
        field = next(iter(rows[0]))
        return {"id": uuid4().hex, "title": field.replace("_", " ").title(), "description": "Live tenant-scoped query", "chart_type": "kpi", "data": rows, "sql": sql.replace(":tenant_id", "[server-bound tenant]"), "value_field": field, "y_fields": [], "number_format": "compact", "layout": {"col_span": 3, "row_span": 1}}
    fields = list(rows[0]) if rows else []
    numeric = next((key for key in fields if rows and isinstance(rows[0].get(key), (int, float))), None)
    chart_type = "bar" if len(fields) >= 2 and numeric else "table"
    return {"id": uuid4().hex, "title": "Query result", "description": "Live tenant-scoped query", "chart_type": chart_type, "data": rows, "sql": sql.replace(":tenant_id", "[server-bound tenant]"), "x_field": fields[0] if chart_type == "bar" else None, "y_fields": [numeric] if chart_type == "bar" and numeric else [], "number_format": "compact", "layout": {"col_span": 6, "row_span": 2}}


def run_enterprise_dag(actor: Actor, question: str) -> dict:
    if not get_settings().openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required. The DAG has no heuristic fallback.")
    catalog = _catalog_text(actor)

    def relevance(state: DAGState) -> DAGState:
        chain = _llm(actor).with_structured_output(RelevanceAndSQL)
        result = chain.invoke(
            "Classify whether this is a business analytics question answerable from these permitted tables. "
            "If relevant, also generate exactly one PostgreSQL SELECT. Use only permitted tables, include "
            "exactly :tenant_id as the tenant filter, never access identities, audit logs or secrets, and return "
            "no prose. If not relevant, set sql_query to null.\n"
            f"Permitted tables:\n{catalog}\nQuestion: {state['question']}"
        )
        return {"relevance": result.relevance, "sql_query": result.sql_query or ""}

    def generate_sql(state: DAGState) -> DAGState:
        # SQL was drafted atomically with relevance above. This explicit DAG
        # node remains a policy boundary before validation/execution.
        if not state.get("sql_query"):
            return {"sql_error": "No SQL draft was produced for a relevant question."}
        return {"sql_query": state["sql_query"]}

    def execute_sql(state: DAGState) -> DAGState:
        try:
            sql = _validate_sql(state["sql_query"], actor)
            return {"sql_query": sql, "query_rows": _execute(sql, actor), "sql_error": ""}
        except Exception as exc:
            return {"sql_error": str(exc)}

    def repair(state: DAGState) -> DAGState:
        chain = _llm(actor).with_structured_output(SQLDraft)
        result = chain.invoke(f"Repair this failed PostgreSQL analytics SQL. Use only:\n{catalog}\nKeep :tenant_id. Error: {state['sql_error']}\nSQL: {state['sql_query']}")
        return {"sql_query": result.sql_query, "attempts": state.get("attempts", 0) + 1}

    def answer(state: DAGState) -> DAGState:
        chain = _llm(actor).with_structured_output(Answer)
        result = chain.invoke(f"Answer the question in the user's language using only these SQL rows. Do not mention IDs or SQL. Question: {state['question']}\nRows: {json.dumps(state.get('query_rows', []), default=str)[:12000]}")
        return {"answer": result}

    def unrelated(state: DAGState) -> DAGState:
        return {"answer": Answer(title="Business data copilot", message="Câu hỏi không liên quan đến dữ liệu nghiệp vụ mà vai trò của bạn được phép truy cập.")}

    def sql_failure(state: DAGState) -> DAGState:
        return {"answer": Answer(
            title="Không thể tạo truy vấn an toàn",
            message="Câu hỏi thuộc phạm vi dữ liệu được phép, nhưng hệ thống không thể tạo một truy vấn SQL hợp lệ sau các lần kiểm tra. Hãy thử diễn đạt cụ thể hơn về chỉ số, thời gian hoặc nhóm dữ liệu cần phân tích.",
        )}

    graph = StateGraph(DAGState)
    graph.add_node("check_relevance", relevance); graph.add_node("generate_sql", generate_sql); graph.add_node("execute_sql", execute_sql)
    graph.add_node("repair_sql", repair); graph.add_node("generate_answer", answer); graph.add_node("unrelated", unrelated); graph.add_node("sql_failure", sql_failure)
    graph.set_entry_point("check_relevance")
    graph.add_conditional_edges("check_relevance", lambda s: "generate_sql" if s["relevance"] == "relevant" else "unrelated")
    graph.add_edge("generate_sql", "execute_sql")
    graph.add_conditional_edges("execute_sql", lambda s: "generate_answer" if not s.get("sql_error") else ("repair_sql" if s.get("attempts", 0) < MAX_SQL_REPAIR_ATTEMPTS else "sql_failure"))
    graph.add_edge("repair_sql", "execute_sql"); graph.add_edge("generate_answer", END); graph.add_edge("unrelated", END); graph.add_edge("sql_failure", END)
    result = graph.compile().invoke({"question": question, "attempts": 0})
    answer_result = result["answer"]
    return {"title": answer_result.title, "message": answer_result.message, "mode": "openai", "widgets": [_widget(result.get("query_rows", []), result.get("sql_query", ""))] if result.get("query_rows") is not None else [], "security": {"tenant": actor.email.split("@")[0], "role": actor.role, "prompt_decision": "allowed"}}
