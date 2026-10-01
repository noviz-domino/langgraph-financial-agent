"""웹 서버 — 브라우저 화면(web/index.html)과 에이전트 그래프를 잇는다.

터미널의 main.py와 같은 일을 HTTP로 한다: 말을 받아 그래프를 돌리고, 멈춰 있으면(되묻기·승인) 다음 말을 재개 답으로 보낸다.
그래프·업무 코드는 고치지 않는다 — 입구만 다르다.

공개할 때를 위한 장치 (2026-09-29, feat/web)
    방문자마다 장부 따로   세션마다 data/ledgers/<세션>.json. "처음으로"로 원본 상태로
    호출 횟수 제한        세션마다 1분에 3번·10분에 10번, 서버 전체 하루 50번 (Gemini API 비용 보호)
    입력 길이 제한        300자

실행:  uv run python -m uvicorn web:app --app-dir src --port 8000   → http://localhost:8000
"""

import logging
import threading
import time
import uuid
from collections import defaultdict, deque
from datetime import date
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

import data_store
from config import ConfigError, load_env
from functions import CURRENT_USER_ID
from graph import build_graph
from intents import INTENTS

logger = logging.getLogger(__name__)
logging.getLogger("google_genai").setLevel(logging.ERROR)     # AFC 안내 경고 숨김 (main.py와 같음)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
MAX_TEXT = 300
SESSION_LIMITS = [(3, 60), (10, 600)]                          # 세션마다 (몇 번, 몇 초 안에) — 1분에 3번, 10분에 10번
SESSION_WINDOW = max(window for _, window in SESSION_LIMITS)    # 이보다 오래된 요청 시각은 버린다
DAILY_LIMIT = 50                                               # 서버 전체 하루 50번
LEDGER_TTL = 24 * 3600                                         # 하루 지난 방문자 장부는 지운다

# 오류 안내 문장 (HTTP 오류의 detail — 화면이 그대로 보여준다)
SESSION_EXPIRED = "세션이 만료됐어요. 새로고침해 주세요."                       # 404
TOO_MANY_REQUESTS = "요청이 너무 많아요. 잠시 후 다시 시도해 주세요."            # 429 — 방문자 제한
DAILY_LIMIT_REACHED = "오늘 사용량이 다 찼어요. 내일 다시 시도해 주세요."        # 429 — 서버 전체 제한
SERVER_CONFIG_ERROR = "서버 설정 문제로 지금은 답할 수 없어요."                  # 500 — API 키 문제

app = FastAPI(title="은행 업무 도우미")
try:
    load_env()                                                 # 첫 요청 전에 — LangSmith는 Tracing 여부를 처음 한 번만 읽고 기억한다
except ConfigError:
    logger.exception("시작: API 키 설정 문제 (채팅 요청 때 다시 안내)")
agent = build_graph()

_threads: dict[str, str] = {}                                  # 세션 → 지금 대화의 thread_id ("처음으로"면 새로)
_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)   # 같은 세션의 요청이 겹치지 않게 (두 번 클릭 등)
_recent: dict[str, deque] = defaultdict(deque)                 # 세션마다 최근 요청 시각
_daily = {"day": date.today(), "count": 0}


class SessionIn(BaseModel):
    session_id: str = Field(pattern=r"^[0-9a-f]{32}$")


class ChatIn(SessionIn):
    text: str = Field(min_length=1, max_length=MAX_TEXT)


# ── 도우미 ───────────────────────────────────────────────────────
def _config(session_id: str) -> dict:
    if session_id not in _threads:                            # 서버가 다시 켜져서 모르는 세션 → 화면이 새로 만든다
        raise HTTPException(404, SESSION_EXPIRED)
    return {"configurable": {"thread_id": _threads[session_id]}}


def _check_limits(session_id: str) -> None:
    """호출 횟수 제한. 넘으면 429 (Too Many Requests)."""
    now = time.time()
    recent = _recent[session_id]
    while recent and now - recent[0] > SESSION_WINDOW:
        recent.popleft()
    if any(sum(now - t <= window for t in recent) >= limit for limit, window in SESSION_LIMITS):
        raise HTTPException(429, TOO_MANY_REQUESTS)
    if _daily["day"] != date.today():
        _daily.update(day=date.today(), count=0)
    if _daily["count"] >= DAILY_LIMIT:
        raise HTTPException(429, DAILY_LIMIT_REACHED)
    recent.append(now)
    _daily["count"] += 1


def _pending(config: dict):
    """그래프가 멈춰 있으면(되묻기·승인 대기) 그 interrupt 값을, 아니면 None."""
    for task in agent.get_state(config).tasks:
        if task.interrupts:
            return task.interrupts[0].value
    return None


def _ledger() -> dict:
    """화면 왼쪽에 보여줄 장부 — 내 계좌·카드와 최근 처리 기록 5개."""
    data = data_store.load()
    return {
        "accounts": [{"name": a["nickname"], "balance": a["balance"]}
                     for a in data["accounts"] if a["owner_id"] == CURRENT_USER_ID],
        "cards": [{"name": c["name"], "status": c["status"]}
                  for c in data["cards"] if c["owner_id"] == CURRENT_USER_ID],
        "requests": [{"id": r["request_id"], "label": INTENTS[r["intent"]]["label"], "summary": _summary(r, data),
                      "time": r["finished_at"][11:16], "status": r["status"]}
                     for r in data["requests"][-5:]][::-1],
    }


def _summary(request: dict, data: dict) -> str:
    """처리 기록 한 줄 요약. 기록에는 ID만 있으니(별명은 바뀔 수 있음) 지금 이름으로 풀어 쓴다."""
    names = {a["account_id"]: a["nickname"] for a in data["accounts"]} | {c["card_id"]: c["name"] for c in data["cards"]}
    d = request["details"]
    if "amount" in d:
        return f"{names.get(d['from_account_id'], '?')} → {names.get(d['to_account_id'], '?')} {d['amount']:,}원"
    if "card_id" in d:
        return f"{names.get(d['card_id'], '?')} 잠금"
    return INTENTS[request["intent"]]["label"]


def _cleanup_old_ledgers() -> None:
    now = time.time()
    for path in data_store.LEDGER_DIR.glob("*.json"):
        if now - path.stat().st_mtime > LEDGER_TTL:
            path.unlink(missing_ok=True)


# ── API ──────────────────────────────────────────────────────────
@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.post("/api/session")
def new_session():
    """방문자 세션을 만들고 원본에서 장부를 복사한다."""
    _cleanup_old_ledgers()
    session_id = uuid.uuid4().hex
    _threads[session_id] = uuid.uuid4().hex
    with data_store.use_ledger(session_id):
        data_store.reset()
        return {"session_id": session_id, "ledger": _ledger()}


@app.post("/api/reset")
def reset(body: SessionIn):
    """"처음으로" — 장부를 원본으로, 대화도 새로 (승인 대기 중이던 요청도 버린다)."""
    _config(body.session_id)
    with _locks[body.session_id], data_store.use_ledger(body.session_id):
        data_store.reset()
        _threads[body.session_id] = uuid.uuid4().hex
        return {"ledger": _ledger()}


@app.post("/api/chat")
def chat(body: ChatIn):
    """말 하나를 처리한다. 멈춰 있으면 재개 답으로, 아니면 새 요청으로 (main.py의 run_turn과 같음)."""
    config = _config(body.session_id)
    _check_limits(body.session_id)
    with _locks[body.session_id], data_store.use_ledger(body.session_id):
        waiting = _pending(config) is not None
        payload = Command(resume=body.text) if waiting else {"messages": [HumanMessage(content=body.text)]}
        try:
            state = agent.invoke(payload, config)
        except ConfigError:
            logger.exception("chat: API 키 설정 문제")
            raise HTTPException(500, SERVER_CONFIG_ERROR)
        interrupts = state.get("__interrupt__")
        if interrupts:
            value = interrupts[0].value
            reply = {"type": value.get("kind", "ask"), "text": value["prompt"],
                     "candidates": value.get("candidates", [])}
        else:
            reply = {"type": "answer", "text": state["messages"][-1].content, "candidates": []}
        next_nodes = agent.get_state(config).next               # 지금 그래프가 멈춘 노드 (화면에 표시)
        return {**reply, "node": next_nodes[0] if next_nodes else None, "ledger": _ledger()}
