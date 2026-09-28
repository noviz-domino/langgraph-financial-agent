"""State 정의, 노드, 조건부 Edge를 담당한다.

조회 경로 (Step 3):  START → understand → read_task → respond → END
변경 경로 (Step 5):  START → understand → plan_change → confirm_change(🛑 interrupt) → apply_change → respond → END
                     검사 불가·취소·만료면 respond로 바로 간다. 애매한 답이면 confirm_change를 다시 보여준다
실패 기록·rollback은 Step 6, 되묻기(ask_more)·수정 루프는 Step 8.
설계는 docs/설계서.md, 바뀐 이유는 docs/설계변경기록.md와 devlog 참고.

LLM을 부르는 곳은 understand 한 군데뿐이다.
사람의 말이 들어오는 입구에서만 LLM이 번역하고, 안쪽은 전부 코드로 처리한다.

단독 실행:  uv run python src/graph.py   (Step 3~5 완료 기준 확인 — Gemini API를 실제로 호출함)
"""

import logging
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from pydantic import Field, create_model

import data_store
from config import load_env
from functions import HANDLERS, READ_TASKS, josa, my_account_names, my_card_names
from integrity import KST
from intents import INTENTS, NOT_UNDERSTOOD, READ_INTENTS, UNCERTAIN, UNSUPPORTED, WRITE_INTENTS

logger = logging.getLogger(__name__)

MODEL_NAME = "gemini-3.5-flash-lite"
APPROVAL_TTL = timedelta(minutes=5)       # 승인 유효 시간 — 처리안을 만든 뒤 이 시간이 지난 "예"는 실행하지 않는다 (Step 5 결정 4)

# 승인 답 — 글자가 정확히 같을 때만 인정한다 (앞뒤 공백·문장부호는 무시). 그 밖의 답은 다시 묻는다 (Step 5 결정 3)
# 부분 일치를 쓰지 않는 이유: "아니요, 진행하지 마"가 "진행"에 걸려 승인되면 안 된다
APPROVE_WORDS = {"예", "네", "응", "ㅇㅇ", "승인", "진행", "진행해", "진행해줘", "보내", "보내줘", "좋아", "오케이", "네 진행해"}
REJECT_WORDS = {"취소", "아니요", "아니오", "거절", "그만", "안 할래", "하지 마", "취소해", "취소해줘"}

_clock_offset = timedelta(0)              # 테스트용 — 시각을 앞으로 돌려 만료를 흉내 낸다 (시계 주입)


def _now() -> datetime:
    """지금 시각(KST). 노드는 datetime.now 대신 이 함수를 부른다 → 테스트에서 5분을 기다리지 않아도 된다."""
    return datetime.now(KST) + _clock_offset


# ── State ────────────────────────────────────────────────────────
class AgentState(TypedDict, total=False):
    """노드들이 주고받는 데이터. 지금 경로에서 쓰는 칸만 둔다 (승인 관련 칸은 Step 5, 8에서 추가).

    messages  대화 내내 쌓인다 (add_messages: 같은 메시지가 다시 오면 중복으로 쌓지 않고 교체)
              기록용. LLM에게는 마지막 메시지 하나만 넘긴다 (맥락은 아래 intent·params로)
    intent    턴을 넘어 이어진다. understand가 직전 값을 LLM에게 보여주고, 이번 턴의 최종 값을 쓴다
    params    턴을 넘어 이어진다. 유지할지·바꿀지는 LLM이 이번 말의 뜻으로 판단한다 (코드가 합치지 않는다)
    inferred  한 턴짜리. LLM이 뜻으로 추측해서 고른 이름들 (사용자 말에 글자로는 없는 것).
              understand가 쓰고 respond가 읽어서 "'여행 자금' 계좌로 이해했어요"처럼 해석을 밝힌다
    result    한 턴짜리. understand가 비우고 read_task·plan_change·apply_change가 쓴다
    plan      승인 대기 중. plan_change가 쓰고 confirm_change·apply_change가 읽는다 (만든 시각 created_at 포함)
    decision  한 턴짜리. confirm_change가 쓴다 — approve / reject / unclear / expired
    """
    messages: Annotated[list, add_messages]
    intent: str
    params: dict
    inferred: list[dict]
    result: dict | None
    plan: dict | None
    decision: str | None


# ── understand: 사람 말 → intent + slot ──────────────────────────
SYSTEM_PROMPT = """너는 은행 앱의 요청 해석기다. 사용자의 가장 최근 메시지를 읽고
무슨 업무를 원하는지(intent)와 필요한 값(slot)을 정해진 칸에 채운다. 문장으로 대답하지 않는다.

<업무 목록>
{intent_lines}
- {unsupported}: 위 목록으로 처리할 수 없는 요청 (잡담, 날씨, 목록에 없는 은행 업무 등)
</업무 목록>

<사용자의 계좌 이름>
{account_names}
</사용자의 계좌 이름>

<사용자의 카드 이름>
{card_names}
</사용자의 카드 이름>

<이전 상태>
{previous}
</이전 상태>

<규칙>
1. 사용자가 말하지 않은 칸은 비워둔다. 절대 추측해서 채우지 않는다. (아래 7·8번으로 이어받는 경우만 예외)
2. 계좌·카드는 위 목록의 이름 중에서만 고른다. 뜻으로 말해도("여행 갈 때 쓰는 통장") 알맞은 이름을 고른다.
3. 계좌·카드를 말했지만 목록 중 어느 것인지 확신할 수 없으면 {uncertain}을 고른다.
   목록에 없는 종류의 계좌("주식 계좌")라도 계좌에 대한 업무를 원하면 {unsupported}가 아니라 그 업무 + {uncertain}이다.
4. 한 문장에 계좌가 여럿이면 역할을 나눈다. "~에서"는 출금 계좌, "~으로"·"~에"는 입금 계좌.
5. 금액은 원 단위 정수로 바꾼다 ("10만 원" → 100000). 음수여도 바꾸지 말고 그대로 옮긴다.
6. 이전 상태를 보고 "이번 턴의 최종 값"을 통째로 채운다.
7. 같은 업무를 이어가면 이전 값은 유지하고, 이번 말에서 바꾼 칸만 고친다.
   "저축은?"은 대상을 바꾸고, "저축도"는 더하고, "전부"는 계좌 목록을 비운다(= 전체).
8. 다른 업무로 바뀌면 이전 상태의 "이어받을 계좌"만 이번 말이 비워 둔 계좌 칸에 옮긴다.
   ("이어받을 계좌: 생활비"일 때 "저축으로 3만 원" → 출금 계좌 = 생활비)
   "이어받을 계좌: 없음"이면 가리키는 말("거기서", "그 계좌")이 있어도 옮기지 않고 비워 둔다.
9. "두 번째 거"처럼 순서로 고르면 이전 상태의 "보여준 후보" 순서를 따른다.
</규칙>"""


def _carry_account(params: dict) -> str | None:
    """다른 업무로 넘어갈 때 이어받을 계좌 하나를 정한다 (규칙 8). 없거나 여럿이면 None.

    개수 세기는 사실이므로 LLM이 아니라 코드가 한다 — 2026-09-28 LLM이 두 계좌를 "하나뿐"으로 잘못 셈.
    params 예) {"accounts": ["생활비"]}                                  → "생활비"
               {"accounts": ["생활비", "저축"]}                          → None
               {"from_account": "생활비", "to_account": "저축", ...}     → None (두 개 → 되묻기)
    """
    names = {name for slot, name in _names_in(params)
             if slot != "card" and name != UNCERTAIN}               # 카드는 계좌가 아니다
    return names.pop() if len(names) == 1 else None


def _describe_previous(state: AgentState) -> str:
    """직전 턴의 해석 결과를 LLM에게 줄 한 줄로 만든다 (대화 기록 대신 — 설계변경기록 참고).

    understand가 이번 턴 값을 쓰기 "전에" 부르므로, State에는 아직 직전 턴의 값이 남아 있다.
    """
    intent = state.get("intent")
    if intent is None:
        return "없음 (첫 요청)"
    params = state.get("params") or {}
    lines = [f"업무: {intent}", f"값: {params}", f"이어받을 계좌: {_carry_account(params) or '없음'}"]
    candidates = (state.get("result") or {}).get("candidates")
    if candidates:
        lines.append(f"보여준 후보(순서대로): {candidates}")
    return "\n".join(lines)


# 계좌·카드 이름이 들어가는 slot → 해석을 밝힐 때 이름 뒤에 붙일 말
# (카드 이름은 이미 "생활비 카드"처럼 끝에 "카드"가 붙어 있어서 빈칸)
NAME_SLOTS = {"accounts": "계좌", "from_account": "계좌", "to_account": "계좌", "card": ""}


def _names_in(params: dict) -> list[tuple[str, str]]:
    """params 안의 계좌·카드 이름을 (slot, 이름) 목록으로 펼친다."""
    pairs = []
    for slot in NAME_SLOTS:
        value = params.get(slot)
        if value is None:
            continue
        pairs.extend((slot, name) for name in (value if isinstance(value, list) else [value]))
    return pairs


def _find_inferred(params: dict, utterance: str, previous_params: dict) -> list[dict]:
    """사용자가 글자로 말하지 않았는데 들어간 이름을 찾는다. 막기 위한 게 아니라 "어떻게 이해했는지 알려주기" 위한 것.

    how="guessed"  이전 상태에도 없던 이름 = 뜻으로 추측 ("여행 갈 때 쓰는 통장" → 여행 자금)
    how="carried"  이전 상태의 다른 칸에서 옮겨 온 이름 ("생활비" 조회 후 "저축으로 3만 원" → 출금 = 생활비)
    같은 칸에서 그대로 유지된 이름은 앞 턴에서 이미 확인했으므로 넣지 않는다 ("아니 5만 원").
    띄어쓰기는 무시하고 비교한다 ("여행자금"이라고 말했으면 "여행 자금"은 추측이 아님).
    """
    said = utterance.replace(" ", "")
    previous_pairs = set(_names_in(previous_params))
    previous_names = {name for _, name in previous_pairs}
    found = []
    for slot, name in _names_in(params):
        if name == UNCERTAIN or name.replace(" ", "") in said or (slot, name) in previous_pairs:
            continue
        found.append({"slot": slot, "name": name, "how": "carried" if name in previous_names else "guessed"})
    return found


SLOT_ROLES = {"accounts": "조회 계좌", "from_account": "출금 계좌", "to_account": "입금 계좌", "card": "카드"}


def _interpretation_notes(inferred: list[dict]) -> str:
    """해석 안내 문장. 추측·이어받음이 없으면 빈 문자열 (필요할 때만 보여준다 — 승인 피로 대비)."""
    lines = []
    guessed = [i for i in inferred if i["how"] == "guessed"]
    if guessed:
        phrases = ", ".join(f"'{i['name']}' {NAME_SLOTS[i['slot']]}".strip() for i in guessed)
        lines.append(f"{phrases}{josa(phrases, '으로/로')} 이해했어요.")
    for i in inferred:
        if i["how"] == "carried":
            role, name = SLOT_ROLES[i["slot"]], i["name"]
            lines.append(f"{role}{josa(role, '은/는')} 방금 보신 '{name}'{josa(name, '으로/로')} 했어요.")
    return "\n".join(lines)


@lru_cache(maxsize=1)
def _get_llm() -> ChatGoogleGenerativeAI:
    """Gemini 모델을 처음 필요할 때 한 번만 만든다 (lru_cache가 두 번째부터는 만들어둔 것을 돌려줌).

    만들기 전에 API 키를 불러온다. 키가 없으면 여기서 ConfigError — 시작할 때 바로 알린다 (fail fast).
    """
    load_env()
    return ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0)   # temperature 0: 같은 입력에 최대한 같은 답


def _build_schema(account_names: list[str], card_names: list[str]):
    """LLM이 채울 칸의 모양(Pydantic 모델)을 실행할 때마다 만든다.

    intent 목록은 intents.py(코드)에서, 계좌·카드 이름은 데이터에서 온다.
    이름은 사용자마다 다르고 별명을 바꾸면 달라지므로 코드에 고정할 수 없다 → create_model로 그때그때 만든다.
    각 칸의 description은 LLM에게 그대로 전달되는 지시문이다.
    """
    IntentName = Literal[tuple([*INTENTS, UNSUPPORTED])]       # 허용값 목록 → 이 밖의 값은 형식 위반
    AccountName = Literal[tuple([*account_names, UNCERTAIN])]
    CardName = Literal[tuple([*card_names, UNCERTAIN])]

    return create_model(
        "Understanding",
        intent=(IntentName, Field(description="사용자가 원하는 업무")),
        # 모든 칸은 "이번 턴의 최종 값"이다 — 이전 상태에서 유지한 값도 다시 적는다 (규칙 6~8)
        accounts=(list[AccountName], Field(default_factory=list,
                  description="이번 턴의 최종 조회 대상. 이전 대상을 유지하면 다시 적는다. 전체면 빈 목록")),
        from_account=(AccountName | None, Field(default=None,
                      description="이체의 출금 계좌('~에서'). 이전 상태에도, 이번 말에도 없으면 비움")),
        to_account=(AccountName | None, Field(default=None,
                    description="이체의 입금 계좌('~으로'). 이전 상태에도, 이번 말에도 없으면 비움")),
        amount=(int | None, Field(default=None,
                description="금액. 원 단위 정수('10만 원'은 100000). 음수도 그대로. 이전 상태에도, 이번 말에도 없으면 비움")),
        card=(CardName | None, Field(default=None,
              description="대상 카드 이름. 이전 상태에도, 이번 말에도 없으면 비움")),
        reason=(str, Field(description="판단 근거 한 문장")),
    )


# 한 턴짜리 칸 — understand가 매 턴 비워서 이전 턴 값이 남지 않게 한다
TURN_RESET = {"result": None, "plan": None, "decision": None}


def understand(state: AgentState) -> dict:
    """사용자 말을 intent와 slot으로 바꾼다. LLM을 부르는 유일한 노드."""
    data = data_store.load()
    account_names = my_account_names(data)                     # 내 것만, 이름만 (잔액은 LLM에 넘기지 않음)
    card_names = my_card_names(data)

    schema = _build_schema(account_names, card_names)
    system = SYSTEM_PROMPT.format(
        intent_lines="\n".join(f"- {name}: {info['desc']}" for name, info in INTENTS.items()),
        unsupported=UNSUPPORTED,
        uncertain=UNCERTAIN,
        account_names=", ".join(account_names),
        card_names=", ".join(card_names),
        previous=_describe_previous(state),
    )
    llm = _get_llm().with_structured_output(schema)            # 키가 없으면 여기서 멈춘다 (try 밖에 둔 이유)
    utterance = state["messages"][-1]                           # 대화 기록 대신 지금 말 하나만 (맥락은 <이전 상태>로)
    previous_params = state.get("params") or {}

    try:
        # 외부(API) 경계만 감싼다 — 네트워크 오류, 형식 위반 같은 LLM 쪽 실패.
        # 우리 코드의 버그까지 감싸지 않도록 LLM 호출 한 줄만 try 안에 둔다.
        parsed = llm.invoke([SystemMessage(content=system), utterance])
    except Exception:
        logger.exception("understand: LLM 호출 또는 형식 검증에 실패했습니다")   # 오류와 traceback을 함께 기록 (29번)
        return {"intent": NOT_UNDERSTOOD, "params": {}, "inferred": [], **TURN_RESET}

    if parsed is None:                                          # 구조화 결과가 비어서 오는 경우
        logger.warning("understand: LLM이 빈 결과를 돌려줬습니다")
        return {"intent": NOT_UNDERSTOOD, "params": {}, "inferred": [], **TURN_RESET}

    # 고른 intent에 해당하는 slot만 남기고 나머지 칸은 버린다 (intents.py가 기준)
    info = INTENTS.get(parsed.intent, {})                       # unsupported면 빈 dict → slot 없음
    params = {}
    for name in [*info.get("required", []), *info.get("optional", [])]:
        value = getattr(parsed, name)
        if value in (None, [], ""):
            continue                                            # 비어 있는 칸은 넣지 않음 (말하지 않은 것)
        params[name] = list(dict.fromkeys(value)) if isinstance(value, list) else value   # 목록은 중복 제거

    # 규칙 8을 코드로 강제 — 다른 업무로 넘어가면서 다른 칸에서 옮겨 온 계좌는 코드가 정한 "이어받을 계좌"만 허용.
    # 프롬프트에 "이어받을 계좌: 없음"을 적어줘도 LLM이 옮겨 온 적이 있다 (2026-09-28). 어긴 칸은 비워서 되묻게 한다
    if parsed.intent != state.get("intent"):
        allowed = _carry_account(previous_params)
        for item in _find_inferred(params, utterance.content, previous_params):
            if item["how"] == "carried" and item["name"] != allowed:
                logger.info("understand: 이어받을 수 없는 계좌를 비움 slot=%s name=%s", item["slot"], item["name"])
                value = params[item["slot"]]
                if isinstance(value, list):
                    value.remove(item["name"])
                if not isinstance(value, list) or not value:
                    params.pop(item["slot"])

    inferred = _find_inferred(params, utterance.content, previous_params)
    logger.info("understand: intent=%s params=%s inferred=%s reason=%s",
                parsed.intent, params, [i["name"] for i in inferred], parsed.reason)
    return {"intent": parsed.intent, "params": params, "inferred": inferred, **TURN_RESET}


# ── 공통: 형식 검사 (intents.py 기준, 업무와 무관) ────────────────
def _form_problem(intent: str, params: dict, data: dict) -> dict | None:
    """필수 slot이 비었거나 uncertain이면 결과(result)를, 문제없으면 None을 돌려준다.

    어느 업무든 똑같은 검사라 Handler가 아니라 그래프가 한다. Step 8에서 되묻기(ask_more)로 바뀐다.
    """
    if any(value == UNCERTAIN or (isinstance(value, list) and UNCERTAIN in value) for value in params.values()):
        return {"ok": False, "notice": "uncertain", "candidates": my_account_names(data)}
    missing = [slot for slot in INTENTS[intent]["required"] if slot not in params]
    if missing:
        return {"ok": False, "notice": "missing", "slots": missing}
    return None


# ── 조건부 Edge ──────────────────────────────────────────────────
# 노드가 아니라 방향만 정하는 함수들이라 State를 바꾸지 않는다.
def route_request(state: AgentState) -> Literal["read_task", "plan_change", "respond"]:
    """조회 → read_task / Handler가 있는 변경 → plan_change / 그 외(처리 불가·이해 실패·준비 중) → respond."""
    intent = state["intent"]
    if intent in READ_INTENTS:
        return "read_task"
    if intent in HANDLERS:
        return "plan_change"
    return "respond"


def route_validation(state: AgentState) -> Literal["confirm_change", "respond"]:
    """처리안이 만들어졌으면 승인 받으러, 아니면(형식 문제·업무상 거절) 바로 응답으로."""
    return "confirm_change" if state.get("plan") else "respond"


def route_decision(state: AgentState) -> Literal["apply_change", "confirm_change", "respond"]:
    """승인 → 실행 / 애매한 답 → 승인 화면을 다시 / 거절·만료 → 응답."""
    decision = state["decision"]
    if decision == "approve":
        return "apply_change"
    if decision == "unclear":
        return "confirm_change"
    return "respond"


# ── read_task: 조회 실행 (읽기만) ────────────────────────────────
def read_task(state: AgentState) -> dict:
    """조회 업무를 실행한다. 어떤 함수를 부를지는 READ_TASKS 표에서 찾는다."""
    intent, params = state["intent"], state.get("params", {})
    data = data_store.load()
    problem = _form_problem(intent, params, data)
    if problem:
        return {"result": problem}
    task = READ_TASKS[intent]
    return {"result": {"ok": True, "data": task["run"](data, **params)}}


# ── plan_change: 처리안 만들기 (아직 아무것도 바꾸지 않는다) ─────
def plan_change(state: AgentState) -> dict:
    """형식 검사 → Handler의 업무 검사 → 처리안. 데이터는 읽기만 한다 (설계 원칙 3: 승인 전에는 바꾸지 않는다)."""
    intent, params = state["intent"], state.get("params", {})
    data = data_store.load()
    problem = _form_problem(intent, params, data)
    if problem:
        return {"result": problem}

    plan, reason = HANDLERS[intent].validate(params, data)
    if reason:
        return {"result": {"ok": False, "notice": "rejected", "reason": reason}}
    return {"plan": {**plan, "created_at": _now().isoformat()}}   # 만든 시각 — 승인 유효 시간의 기준


# ── confirm_change: 승인 받기 (🛑 interrupt) ────────────────────
def _classify(answer: str) -> str:
    """승인 답을 approve / reject / unclear로. 정해진 단어와 글자가 정확히 같을 때만 인정한다 (LLM 안 씀)."""
    text = answer.strip().strip(".!?~ ").strip()
    if text in APPROVE_WORDS:
        return "approve"
    if text in REJECT_WORDS:
        return "reject"
    return "unclear"


def confirm_change(state: AgentState) -> dict:
    """승인 화면을 보여주고 멈춘다. 재개(Command(resume=답))되면 답을 분류한다.

    interrupt는 재개될 때 이 노드를 처음부터 다시 실행하고, interrupt(...) 자리에서 사용자의 답을 돌려준다.
    그래서 interrupt 앞에는 부작용(저장 등)이 없어야 한다 — 여기는 문장을 만들 뿐이다.
    """
    plan = state["plan"]
    lines = []
    if state.get("decision") == "unclear":                     # 방금 애매한 답을 받고 다시 보여주는 중
        lines.append("'예' 또는 '취소'로 답해 주세요.")
    notes = _interpretation_notes(state.get("inferred") or [])
    if notes:
        lines.append(notes)
    lines.append(HANDLERS[state["intent"]].describe(plan))
    lines.append("진행하려면 '예', 그만두려면 '취소'라고 말씀해 주세요.")

    answer = interrupt({"prompt": "\n".join(lines)})           # 🛑 여기서 멈춘다. main.py가 답을 받아 재개한다

    decision = _classify(answer)
    if decision == "approve" and _now() - datetime.fromisoformat(plan["created_at"]) > APPROVAL_TTL:
        decision = "expired"                                   # 방치했다 돌아와서 누른 "예"는 실행하지 않는다
    logger.info("confirm_change: answer=%r decision=%s", answer, decision)
    return {"decision": decision}


# ── apply_change: 실행 직전 재검사 → 실행 → 저장 ────────────────
def apply_change(state: AgentState) -> dict:
    """장부를 새로 읽어 처리안을 만들 때와 **같은 validate**로 다시 검사한 뒤 실행하고 저장한다 (설계 원칙 4).

    승인을 기다리는 사이 잔액이 바뀌었을 수 있다 (TOCTOU). 저장 실패 처리(rollback·기록)는 Step 6.
    """
    handler = HANDLERS[state["intent"]]
    data = data_store.load()
    plan, reason = handler.validate(state["params"], data)
    if reason:
        return {"result": {"ok": False, "notice": "rejected",
                           "reason": f"승인하시는 사이 계좌 상황이 바뀌어 실행하지 않았어요.\n{reason}"},
                "plan": None}

    message = handler.apply(plan, data, _now())
    data_store.save(data)                                       # 무결성 검사(세 번째 안전망) + atomic write
    return {"result": {"ok": True, "message": message}, "plan": None}


# ── respond: 결과 → 문장 (정해진 틀) ─────────────────────────────
def respond(state: AgentState) -> dict:
    """결과를 사람이 읽을 문장으로 만든다. 모든 경로가 여기로 모인다. LLM은 쓰지 않는다."""
    intent = state.get("intent")
    result = state.get("result") or {}
    decision = state.get("decision")

    if intent == UNSUPPORTED:
        labels = ", ".join(info["label"] for info in INTENTS.values())
        text = f"죄송해요, 그 요청은 도와드릴 수 없어요. 지금 할 수 있는 일: {labels}"
    elif intent == NOT_UNDERSTOOD:
        text = "요청을 이해하지 못했어요. 다시 말씀해 주세요."
    elif intent in WRITE_INTENTS and intent not in HANDLERS:
        text = f"{INTENTS[intent]['label']} 기능은 아직 준비 중이에요."          # Step 7에서 카드 잠금
    elif decision == "reject":
        label = INTENTS[intent]["label"]
        text = f"{label}{josa(label, '을/를')} 취소했어요. 아무것도 바뀌지 않았어요."
    elif decision == "expired":
        minutes = int(APPROVAL_TTL.total_seconds() // 60)
        text = f"승인 시간({minutes}분)이 지나 안전을 위해 실행하지 않았어요. 다시 요청해 주세요."
    elif result.get("notice") == "uncertain":
        text = "말씀하신 계좌를 찾지 못했어요. 이 중에서 골라 다시 말씀해 주세요: " + ", ".join(result["candidates"])
    elif result.get("notice") == "missing":
        needed = ", ".join(SLOT_ROLES.get(s, "금액") for s in result["slots"])
        text = f"{needed}{josa(needed, '을/를')} 알려 주세요."
    elif result.get("notice") == "rejected":
        text = result["reason"]
    elif intent in READ_INTENTS:
        text = READ_TASKS[intent]["format"](result["data"])                      # 업무별 틀은 functions.py에
        notes = _interpretation_notes(state.get("inferred") or [])
        if notes:                                                # 뜻으로 추측했으면 어떻게 이해했는지 먼저 밝힌다
            text = f"{notes}\n{text}"
    else:
        text = result["message"]                                 # 변경 완료 — Handler.apply가 만든 문장

    return {"messages": [AIMessage(content=text)]}


# ── 그래프 조립 ──────────────────────────────────────────────────
def build_graph(checkpointer=None):
    """노드와 Edge를 등록하고 Checkpointer와 함께 compile한다."""
    missing = READ_INTENTS - READ_TASKS.keys()                  # intents.py와 functions.py의 약속 확인
    if missing:
        raise RuntimeError(f"READ_TASKS에 조회 함수가 없는 intent가 있습니다: {sorted(missing)}")
    unknown = HANDLERS.keys() - WRITE_INTENTS
    if unknown:
        raise RuntimeError(f"HANDLERS에 변경 intent가 아닌 키가 있습니다: {sorted(unknown)}")

    builder = StateGraph(AgentState)
    for name, node in [("understand", understand), ("read_task", read_task), ("plan_change", plan_change),
                       ("confirm_change", confirm_change), ("apply_change", apply_change), ("respond", respond)]:
        builder.add_node(name, node)                            # 네모 = 노드

    builder.add_edge(START, "understand")                       # START·END는 노드가 아니라 표식
    builder.add_conditional_edges("understand", route_request)  # 마름모 = 조건부 Edge (갈 곳은 Literal 반환형에서 읽음)
    builder.add_edge("read_task", "respond")
    builder.add_conditional_edges("plan_change", route_validation)
    builder.add_conditional_edges("confirm_change", route_decision)
    builder.add_edge("apply_change", "respond")
    builder.add_edge("respond", END)

    return builder.compile(checkpointer=checkpointer or InMemorySaver())


# ── 단독 실행: Step 3~5 완료 기준 확인 (Gemini API 호출 27회) ─────
def _self_check() -> None:
    """구현계획 Step 3·4·5의 완료 기준을 확인한다.

    이어 묻기 확인은 한 대화로, 나머지는 질문마다 새 대화로 나눈다.
    (한 대화로 모두 물으면 앞 질문이 뒤 질문의 해석에 영향을 줄 수 있어서, 무엇을 확인하는지 흐려진다)
    """
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")
    logging.getLogger(__name__).setLevel(logging.INFO)          # 이 모듈의 판단 근거 로그만 보이게 (라이브러리 로그는 경고 이상만)
    graph = build_graph()
    results = []

    nodes = set(graph.get_graph().nodes) - {"__start__", "__end__"}
    results.append(("노드 6개 (조회 경로 + 승인 경로)",
                    nodes == {"understand", "read_task", "plan_change", "confirm_change", "apply_change", "respond"}))

    def ask(text: str, thread_id: str):
        config = {"configurable": {"thread_id": thread_id}}    # thread_id가 같으면 같은 대화 (Checkpointer가 기억)
        state = graph.invoke({"messages": [HumanMessage(content=text)]}, config)
        return state, state["messages"][-1].content

    def record(question: str, state: dict, answer: str, passed: bool):
        results.append((f"{question} -> {' / '.join(answer.splitlines()[:2])}", passed))

    # ① 이어 묻기 확인 — 한 대화 안에서. 이전 상태를 유지·추가·변경·전체로 바꾸는지 (Step 4, B안)
    def accounts_of(state):
        return set(state.get("params", {}).get("accounts", []))

    continuity = [
        ("생활비랑 저축 잔액 보여줘", lambda s, a: accounts_of(s) == {"생활비", "저축"}),
        ("여행 자금도",             lambda s, a: accounts_of(s) == {"생활비", "저축", "여행 자금"}),   # 추가
        ("비상금은?",               lambda s, a: accounts_of(s) == {"비상금"}),                    # 대상 변경
        ("잔액 다시 알려줘",         lambda s, a: accounts_of(s) == {"비상금"}),                    # 말 안 함 = 유지 (전체 아님)
        ("내 계좌 전부 보여줘",      lambda s, a: all(n in a for n in ["생활비", "저축", "여행 자금", "비상금"])),
    ]
    for question, check in continuity:
        state, answer = ask(question, "continuity")
        record(question, state, answer, check(state, answer))

    # ② 업무가 바뀔 때 — 직전 계좌가 하나면 이어받고, 여럿이면 비운다 (Step 4 결정 1-2)
    state, answer = ask("생활비 잔액 얼마야?", "switch-pointed")
    state, answer = ask("거기서 저축으로 3만 원 보내줘", "switch-pointed")
    record("(생활비 조회 후) 거기서 저축으로 3만 원", state, answer,
           state.get("params") == {"from_account": "생활비", "to_account": "저축", "amount": 30000})
    state, answer = ask("아니 5만 원", "switch-pointed")                                       # 같은 업무 → 금액만
    record("아니 5만 원", state, answer,
           state.get("params") == {"from_account": "생활비", "to_account": "저축", "amount": 50000})

    state, answer = ask("생활비 잔액 얼마야?", "switch-silent")
    state, answer = ask("저축으로 3만 원 보내줘", "switch-silent")
    record("(생활비 조회 후) 저축으로 3만 원 -> 출금 = 생활비", state, answer,
           state.get("params", {}).get("from_account") == "생활비")

    state, answer = ask("생활비랑 저축 잔액 보여줘", "switch-many")
    state, answer = ask("여행 자금으로 3만 원 보내줘", "switch-many")
    record("(두 계좌 조회 후) 여행 자금으로 3만 원 -> 출금 비움", state, answer,
           "from_account" not in state.get("params", {}))

    # ③ 보여준 후보를 순서로 고르기
    state, answer = ask("주식 계좌 잔액 알려줘", "candidates")
    candidates = (state.get("result") or {}).get("candidates", [])
    state, answer = ask("두 번째 거", "candidates")
    record("(후보를 본 뒤) 두 번째 거", state, answer,
           len(candidates) > 1 and accounts_of(state) == {candidates[1]})

    # ② 개별 질문 확인 — 질문마다 새 대화. 앞 질문의 영향을 받지 않게
    cases = [
        ("생활비 잔액 얼마야?",                                  # 글자 그대로 말함 → 해석 안내 없이 바로 답
         lambda s, a: a == "생활비 계좌 잔액은 520,000원입니다."),
        ("여행 갈 때 쓰는 통장 잔액 알려줘",                      # 뜻으로만 찾을 수 있음 → 찾되 해석을 밝혀야 함
         lambda s, a: "'여행 자금' 계좌로 이해했어요." in a and "310,000원" in a),
        ("오늘 날씨 어때?",
         lambda s, a: s.get("intent") == UNSUPPORTED),
        ("휴가비 계좌 잔액 알려줘",                                # 없는 이름 → uncertain이거나, 추측했다면 해석을 밝혀야 함
         lambda s, a: (bool(s.get("result")) and not s["result"]["ok"]) or "이해했어요" in a),
        ("주식 계좌 잔액 알려줘",                                  # 뜻이 통하는 계좌가 없음 → uncertain
         lambda s, a: bool(s.get("result")) and not s["result"]["ok"]),
    ]
    for number, (question, check) in enumerate(cases):
        state, answer = ask(question, f"single-{number}")
        record(question, state, answer, check(state, answer))

    # ④ Step 5 — 즉시이체 + 승인 흐름. 작업용 복사본을 원본으로 되돌리고 시작해서, 끝나면 다시 되돌린다
    global _clock_offset
    from functions import _next_id                              # 확인할 때만 필요해서 여기서 불러온다
    from integrity import check_integrity
    data_store.reset()

    def resume(text: str, thread_id: str):
        config = {"configurable": {"thread_id": thread_id}}
        state = graph.invoke(Command(resume=text), config)
        return state, state["messages"][-1].content

    def waiting_prompt(state) -> str:                           # 승인 화면에서 멈췄으면 그 문장, 아니면 ""
        interrupts = state.get("__interrupt__")
        return interrupts[0].value["prompt"] if interrupts else ""

    def balance(name: str) -> int:
        return next(a["balance"] for a in data_store.load()["accounts"]
                    if a["owner_id"] == "user_01" and a["nickname"] == name)

    before = data_store.WORKING_PATH.read_bytes()
    state, _ = ask("생활비에서 저축으로 10만 원 보내줘", "t-approve")
    prompt = waiting_prompt(state)
    record("이체 요청 -> 승인 화면에서 멈춤", state, prompt,
           "100,000원을 즉시이체" in prompt and "420,000원" in prompt)
    results.append(("승인 전에는 장부가 그대로", data_store.WORKING_PATH.read_bytes() == before))
    state, answer = resume("예", "t-approve")
    record("'예' -> 실행", state, answer, balance("생활비") == 420000 and balance("저축") == 1950000)
    results.append(("실행 후 무결성 9개 규칙 통과", not any(check_integrity(data_store.load()).values())))

    before = data_store.WORKING_PATH.read_bytes()
    ask("생활비에서 저축으로 10만 원 보내줘", "t-reject")
    state, answer = resume("취소", "t-reject")
    record("'취소' -> 장부 그대로", state, answer,
           "취소했어요" in answer and data_store.WORKING_PATH.read_bytes() == before)

    ask("생활비에서 저축으로 10만 원 보내줘", "t-unclear")
    state, _ = resume("음 잠깐만", "t-unclear")
    prompt = waiting_prompt(state)
    record("애매한 답 -> 승인 화면을 다시", state, prompt, prompt.startswith("'예' 또는 '취소'로 답해 주세요."))
    state, answer = resume("취소", "t-unclear")
    results.append(("다시 보여준 뒤 '취소' -> 끝남", "취소했어요" in answer))

    before = data_store.WORKING_PATH.read_bytes()
    ask("생활비에서 저축으로 10만 원 보내줘", "t-expired")
    _clock_offset = APPROVAL_TTL + timedelta(minutes=1)         # 6분 뒤에 "예"를 누른 것처럼
    state, answer = resume("예", "t-expired")
    _clock_offset = timedelta(0)
    record("5분 지난 '예' -> 실행 안 함", state, answer,
           "승인 시간" in answer and data_store.WORKING_PATH.read_bytes() == before)

    state, answer = ask("비상금에서 저축으로 100만 원 보내줘", "t-insufficient")
    record("잔액 부족 -> 승인 화면 없이 거절 사유", state, answer,
           not waiting_prompt(state) and "잔액이 부족해요" in answer)

    ask("생활비 잔액 얼마야?", "t-carried")
    state, _ = ask("저축으로 3만 원 보내줘", "t-carried")
    prompt = waiting_prompt(state)
    record("(생활비 조회 후) 저축으로 3만 원 -> 이어받은 계좌를 밝힘", state, prompt,
           prompt.startswith("출금 계좌는 방금 보신 '생활비'"))
    resume("취소", "t-carried")

    ask("비상금에서 저축으로 7만 원 보내줘", "t-changed")      # 비상금 75,000원
    data = data_store.load()                                    # 승인을 기다리는 사이 다른 곳에서 1만 원 출금
    data["transactions"].append({
        "transaction_id": _next_id(data["transactions"], "transaction_id", "tx_"), "owner_id": "user_01",
        "account_id": "acc_004", "type": "withdrawal", "amount": 10000,
        "occurred_at": _now().isoformat(timespec="seconds"), "card_id": None, "merchant": "확인용", "transfer_id": None,
    })
    next(a for a in data["accounts"] if a["account_id"] == "acc_004")["balance"] -= 10000
    data_store.save(data)
    state, answer = resume("예", "t-changed")
    record("승인 사이 잔액 감소 -> 실행 직전 재검사로 거절", state, answer,
           "승인하시는 사이" in answer and balance("비상금") == 65000)

    data_store.reset()

    for description, passed in results:
        print(f"[{'OK  ' if passed else 'FAIL'}] {description}")
    passed_count = sum(1 for _, passed in results if passed)
    print(f"결과: {passed_count}/{len(results)} 통과")


if __name__ == "__main__":   # 직접 실행했을 때만 확인 코드를 돌린다
    _self_check()
