"""State 정의, 노드, 조건부 Edge를 담당한다.

Step 3 (지금): 조회 경로만 — START → understand → route_request → read_task → respond → END
Step 5 이후:  plan_change, confirm_change 등 승인 경로 노드가 추가된다.
설계는 docs/설계서.md, 바뀐 이유는 docs/설계변경기록.md와 devlog 참고.

LLM을 부르는 곳은 understand 한 군데뿐이다.
사람의 말이 들어오는 입구에서만 LLM이 번역하고, 안쪽은 전부 코드로 처리한다.

단독 실행:  uv run python src/graph.py   (Step 3 완료 기준 확인 — Gemini API를 실제로 호출함)
"""

import logging
from functools import lru_cache
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import Field, create_model

import data_store
from config import load_env
from functions import READ_TASKS, my_account_names, my_card_names
from intents import INTENTS, NOT_UNDERSTOOD, READ_INTENTS, UNCERTAIN, UNSUPPORTED, WRITE_INTENTS

logger = logging.getLogger(__name__)

MODEL_NAME = "gemini-3.5-flash-lite"


# ── State ────────────────────────────────────────────────────────
class AgentState(TypedDict, total=False):
    """노드들이 주고받는 데이터. 지금 경로에서 쓰는 칸만 둔다 (승인 관련 칸은 Step 5, 8에서 추가).

    messages  대화 내내 쌓인다 (add_messages: 같은 메시지가 다시 오면 중복으로 쌓지 않고 교체)
              기록용. LLM에게는 마지막 메시지 하나만 넘긴다 (맥락은 아래 intent·params로)
    intent    턴을 넘어 이어진다. understand가 직전 값을 LLM에게 보여주고, 이번 턴의 최종 값을 쓴다
    params    턴을 넘어 이어진다. 유지할지·바꿀지는 LLM이 이번 말의 뜻으로 판단한다 (코드가 합치지 않는다)
    inferred  한 턴짜리. LLM이 뜻으로 추측해서 고른 이름들 (사용자 말에 글자로는 없는 것).
              understand가 쓰고 respond가 읽어서 "'여행 자금' 계좌로 이해했어요"처럼 해석을 밝힌다
    result    한 턴짜리. understand가 비우고 read_task가 쓴다
    """
    messages: Annotated[list, add_messages]
    intent: str
    params: dict
    inferred: list[dict]
    result: dict | None


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
    """LLM이 고른 이름 중, 사용자 말에 글자로 없고 이전 상태에도 없던 것을 찾는다 = 뜻으로 추측한 것.

    띄어쓰기는 무시하고 비교한다 ("여행자금"이라고 말했으면 "여행 자금"은 추측이 아님).
    이전 상태에서 이어받은 이름은 추측이 아니다 (이미 앞 턴에서 확인된 이름).
    이 확인은 막기 위한 것이 아니라 "어떻게 이해했는지 알려주기" 위한 것이다.
    """
    said = utterance.replace(" ", "")
    known = {name for _, name in _names_in(previous_params)}
    return [{"slot": slot, "name": name} for slot, name in _names_in(params)
            if name != UNCERTAIN and name not in known and name.replace(" ", "") not in said]


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
        return {"intent": NOT_UNDERSTOOD, "params": {}, "inferred": [], "result": None}

    if parsed is None:                                          # 구조화 결과가 비어서 오는 경우
        logger.warning("understand: LLM이 빈 결과를 돌려줬습니다")
        return {"intent": NOT_UNDERSTOOD, "params": {}, "inferred": [], "result": None}

    # 고른 intent에 해당하는 slot만 남기고 나머지 칸은 버린다 (intents.py가 기준)
    info = INTENTS.get(parsed.intent, {})                       # unsupported면 빈 dict → slot 없음
    params = {}
    for name in [*info.get("required", []), *info.get("optional", [])]:
        value = getattr(parsed, name)
        if value in (None, [], ""):
            continue                                            # 비어 있는 칸은 넣지 않음 (말하지 않은 것)
        params[name] = list(dict.fromkeys(value)) if isinstance(value, list) else value   # 목록은 중복 제거

    inferred = _find_inferred(params, utterance.content, previous_params)
    logger.info("understand: intent=%s params=%s inferred=%s reason=%s",
                parsed.intent, params, [i["name"] for i in inferred], parsed.reason)
    return {"intent": parsed.intent, "params": params, "inferred": inferred,
            "result": None}                                      # result를 비워 이전 턴 값이 남지 않게


# ── 조건부 Edge ──────────────────────────────────────────────────
def route_request(state: AgentState) -> Literal["read_task", "respond"]:
    """조회 intent면 read_task로, 그 외(변경·처리 불가·이해 실패)는 respond로.

    노드가 아니라 방향만 정하는 함수라 State를 바꾸지 않는다. Step 5부터 변경 intent는 plan_change로 간다.
    """
    return "read_task" if state["intent"] in READ_INTENTS else "respond"


# ── read_task: 조회 실행 (읽기만) ────────────────────────────────
def read_task(state: AgentState) -> dict:
    """조회 업무를 실행한다. 어떤 함수를 부를지는 READ_TASKS 표에서 찾는다."""
    intent, params = state["intent"], state.get("params", {})
    data = data_store.load()

    uncertain_slots = [name for name, value in params.items()
                       if value == UNCERTAIN or (isinstance(value, list) and UNCERTAIN in value)]
    if uncertain_slots:
        # Step 3 임시: 되묻기(ask_more, 루프 1)는 Step 8. 지금은 후보를 보여주며 다시 말해달라고 안내한다
        return {"result": {"ok": False, "notice": "uncertain", "candidates": my_account_names(data)}}

    task = READ_TASKS[intent]
    return {"result": {"ok": True, "data": task["run"](data, **params)}}


# ── respond: 결과 → 문장 (정해진 틀) ─────────────────────────────
def respond(state: AgentState) -> dict:
    """결과를 사람이 읽을 문장으로 만든다. 모든 경로가 여기로 모인다. LLM은 쓰지 않는다."""
    intent = state.get("intent")
    result = state.get("result") or {}

    if intent == UNSUPPORTED:
        labels = ", ".join(info["label"] for info in INTENTS.values())
        text = f"죄송해요, 그 요청은 도와드릴 수 없어요. 지금 할 수 있는 일: {labels}"
    elif intent == NOT_UNDERSTOOD:
        text = "요청을 이해하지 못했어요. 다시 말씀해 주세요."
    elif intent in WRITE_INTENTS:
        text = f"{INTENTS[intent]['label']} 기능은 아직 준비 중이에요."          # Step 3 임시 (Step 5부터 실제 경로)
    elif result.get("ok"):
        text = READ_TASKS[intent]["format"](result["data"])                      # 업무별 틀은 functions.py에
        inferred = state.get("inferred") or []
        if inferred:                                             # 뜻으로 추측했으면 어떻게 이해했는지 먼저 밝힌다
            phrases = ", ".join(f"'{i['name']}' {NAME_SLOTS[i['slot']]}".strip() for i in inferred)
            text = f"{phrases}로 이해했어요.\n{text}"
    else:
        text = "말씀하신 계좌를 찾지 못했어요. 이 중에서 골라 다시 말씀해 주세요: " + ", ".join(result["candidates"])

    return {"messages": [AIMessage(content=text)]}


# ── 그래프 조립 ──────────────────────────────────────────────────
def build_graph(checkpointer=None):
    """노드와 Edge를 등록하고 Checkpointer와 함께 compile한다."""
    missing = READ_INTENTS - READ_TASKS.keys()                  # intents.py와 functions.py의 약속 확인
    if missing:
        raise RuntimeError(f"READ_TASKS에 조회 함수가 없는 intent가 있습니다: {sorted(missing)}")

    builder = StateGraph(AgentState)
    builder.add_node("understand", understand)                  # 네모 = 노드
    builder.add_node("read_task", read_task)
    builder.add_node("respond", respond)

    builder.add_edge(START, "understand")                       # START·END는 노드가 아니라 표식
    builder.add_conditional_edges(                              # 마름모 = 조건부 Edge
        "understand", route_request, {"read_task": "read_task", "respond": "respond"}
    )
    builder.add_edge("read_task", "respond")
    builder.add_edge("respond", END)

    return builder.compile(checkpointer=checkpointer or InMemorySaver())


# ── 단독 실행: Step 3·4 완료 기준 확인 (Gemini API 호출 19회) ─────
def _self_check() -> None:
    """구현계획 Step 3의 완료 기준을 확인한다.

    이어 묻기 확인은 한 대화로, 나머지는 질문마다 새 대화로 나눈다.
    (한 대화로 모두 물으면 앞 질문이 뒤 질문의 해석에 영향을 줄 수 있어서, 무엇을 확인하는지 흐려진다)
    """
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")
    logging.getLogger(__name__).setLevel(logging.INFO)          # 이 모듈의 판단 근거 로그만 보이게 (라이브러리 로그는 경고 이상만)
    graph = build_graph()
    results = []

    nodes = set(graph.get_graph().nodes) - {"__start__", "__end__"}
    results.append(("노드가 조회 경로 3개뿐", nodes == {"understand", "read_task", "respond"}))

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

    for description, passed in results:
        print(f"[{'OK  ' if passed else 'FAIL'}] {description}")
    passed_count = sum(1 for _, passed in results if passed)
    print(f"결과: {passed_count}/{len(results)} 통과")


if __name__ == "__main__":   # 직접 실행했을 때만 확인 코드를 돌린다
    _self_check()
