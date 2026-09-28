from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .schemas import MetaResponse
from .secure_demo import assess_prompt, check_rate_limit, dashboard
from .auth import actor_from_bearer, initialize_auth, invite_user, login, register_company, tenant_users


settings = get_settings()

app = FastAPI(title=settings.app_name, version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin, "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    initialize_auth()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


class SecureQueryRequest(BaseModel):
    prompt: str = Field(min_length=3, max_length=1500)


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=255)
    password: str = Field(min_length=8, max_length=200)


class RegisterRequest(LoginRequest):
    company_name: str = Field(min_length=2, max_length=120)
    display_name: str = Field(min_length=2, max_length=120)


class InviteRequest(LoginRequest):
    display_name: str = Field(min_length=2, max_length=120)
    role: str = Field(pattern="^(sales_manager|operations_manager|hr_manager|security_admin|tenant_admin)$")


def actor_payload(actor) -> dict:
    return {"email": actor.email, "role": actor.role, "tenant_id": actor.tenant_id}


@app.post("/api/auth/login")
def auth_login(request: LoginRequest) -> dict:
    try:
        token, actor = login(request.email, request.password)
        return {"access_token": token, "user": actor_payload(actor)}
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


@app.post("/api/auth/register")
def auth_register(request: RegisterRequest) -> dict:
    try:
        token, actor = register_company(request.company_name, request.email, request.display_name, request.password)
        return {"access_token": token, "user": actor_payload(actor)}
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/auth/me")
def auth_me(authorization: str | None = Header(default=None)) -> dict:
    try:
        return actor_payload(actor_from_bearer(authorization))
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


@app.get("/api/admin/users")
def admin_users(authorization: str | None = Header(default=None)) -> list[dict]:
    try:
        return tenant_users(actor_from_bearer(authorization))
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@app.post("/api/admin/users")
def admin_invite(request: InviteRequest, authorization: str | None = Header(default=None)) -> dict:
    try:
        invite_user(actor_from_bearer(authorization), request.email, request.display_name, request.role, request.password)
        return {"status": "invited"}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/secure/query")
def secure_query(request: SecureQueryRequest, authorization: str | None = Header(default=None)) -> dict:
    try:
        actor = actor_from_bearer(authorization)
        allowed, retry_after = check_rate_limit(f"{actor.tenant_id}:{actor.user_id}")
        if not allowed:
            raise HTTPException(status_code=429, detail="Rate limit exceeded", headers={"Retry-After": str(retry_after)})
        safe, reason = assess_prompt(request.prompt)
        if not safe:
            from .secure_demo import audit
            audit(actor, "prompt_injection_detected", "blocked")
            raise HTTPException(status_code=422, detail=reason)
        return dashboard(actor, request.prompt)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/meta", response_model=MetaResponse)
def meta() -> MetaResponse:
    return MetaResponse(
        app_name=settings.app_name,
        model=settings.openai_model,
        copilot_mode="openai",
    )
