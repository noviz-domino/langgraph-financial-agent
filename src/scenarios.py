"""대화 시나리오로 전체 동작을 확인한다 (그래프 + 업무 코드 + 저장이 함께 도는지).

graph.py 한 파일의 확인이 아니라 여러 모듈을 함께 쓰는 확인이라 따로 둔다 (2026-09-28 검수에서 graph.py에서 옮김).
작업용 복사본(bank_data.json)을 원본으로 되돌리고 시작해서, 끝나면 다시 되돌린다.

단독 실행:  uv run python src/scenarios.py   (Gemini API 41회 호출 + 가짜 LLM 시나리오)
"""

import logging
from datetime import timedelta

from langchain_core.messages import HumanMessage
from langgraph.types import Command

import data_store
import graph as g
from functions import _next_id
from integrity import check_integrity
from intents import UNSUPPORTED

results = []                                                    # (확인 내용, 통과 여부)


# ── 도우미 ───────────────────────────────────────────────────────
def shown(state) -> str:
    """사용자가 화면에서 보는 문장 — 멈췄으면(되묻기·승인) 그 질문, 아니면 답."""
    interrupts = state.get("__interrupt__")
    return interrupts[0].value["prompt"] if interrupts else state["messages"][-1].content


def waiting_prompt(state) -> str:
    """멈췄으면 그 질문, 아니면 ""."""
    interrupts = state.get("__interrupt__")
    return interrupts[0].value["prompt"] if interrupts else ""


def record(question: str, state: dict, answer: str, passed: bool) -> None:
    results.append((f"{question} -> {' / '.join(answer.splitlines()[:2])}", passed))


def accounts_of(state) -> set:
    return set(state.get("params", {}).get("accounts", []))


def ledger() -> dict:
    """잔액·거래 (처리 기록 requests는 뺌 — 취소·만료도 기록은 남는다)."""
    return {key: value for key, value in data_store.load().items() if key != "requests"}


def balance(name: str) -> int:
    return next(a["balance"] for a in data_store.load()["accounts"]
                if a["owner_id"] == "user_01" and a["nickname"] == name)


def card_status(name: str) -> str:
    return next(c["status"] for c in data_store.load()["cards"] if c["owner_id"] == "user_01" and c["name"] == name)


class ScriptedLLM:
    """정해진 답을 차례로 내는 가짜 LLM. Exception을 넣어 두면 그 차례에 실패한다 (API 호출 없음)."""

    def __init__(self, answers: list):
        self.answers = list(answers)

    def with_structured_output(self, schema):
        self.schema = schema
        return self

    def invoke(self, messages):
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return self.schema(reason="가짜", **answer)


def run() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")
    logging.getLogger("graph").setLevel(logging.INFO)          # 판단 근거 로그만 보이게 (라이브러리는 경고 이상만)
    app = g.build_graph()

    def ask(text: str, thread_id: str):                         # 새 요청 (멈춰 있던 질문이 있으면 버리고 처음부터)
        config = {"configurable": {"thread_id": thread_id}}    # thread_id가 같으면 같은 대화 (Checkpointer가 기억)
        state = app.invoke({"messages": [HumanMessage(content=text)]}, config)
        return state, shown(state)

    def resume(text: str, thread_id: str):                      # 멈춘 질문(되묻기·승인)에 답하기
        state = app.invoke(Command(resume=text), {"configurable": {"thread_id": thread_id}})
        return state, shown(state)

    nodes = set(app.get_graph().nodes) - {"__start__", "__end__"}
    results.append(("노드 8개 (조회 + 되묻기 + 승인 + 기록)",
                    nodes == {"understand", "ask_more", "read_task", "plan_change", "confirm_change",
                              "apply_change", "record_result", "respond"}))

    # ① 이어 묻기 — 한 대화 안에서. 이전 상태를 유지·추가·변경·전체로 바꾸는지 (Step 4)
    continuity = [
        ("생활비랑 저축 잔액 보여줘", lambda s, a: accounts_of(s) == {"생활비", "저축"}),
        ("여행 자금도",             lambda s, a: accounts_of(s) == {"생활비", "저축", "여행 자금"}),   # 추가
        ("비상금은?",               lambda s, a: accounts_of(s) == {"비상금"}),                    # 대상 변경
        ("잔액 다시 알려줘",         lambda s, a: accounts_of(s) == {"비상금"}),                    # 말 안 함 = 유지
        ("내 계좌 전부 보여줘",      lambda s, a: all(n in a for n in ["생활비", "저축", "여행 자금", "비상금"])),
    ]
    for question, check in continuity:
        state, answer = ask(question, "continuity")
        record(question, state, answer, check(state, answer))

    # ② 업무가 바뀔 때 — 직전 계좌가 하나면 이어받고, 여럿이면 비운다 (Step 4)
    ask("생활비 잔액 얼마야?", "switch-pointed")
    state, answer = ask("거기서 저축으로 3만 원 보내줘", "switch-pointed")
    record("(생활비 조회 후) 거기서 저축으로 3만 원", state, answer,
           state.get("params") == {"from_account": "생활비", "to_account": "저축", "amount": 30000})
    state, answer = ask("아니 5만 원", "switch-pointed")                                       # 같은 업무 → 금액만
    record("아니 5만 원", state, answer,
           state.get("params") == {"from_account": "생활비", "to_account": "저축", "amount": 50000})

    ask("생활비 잔액 얼마야?", "switch-silent")
    state, answer = ask("저축으로 3만 원 보내줘", "switch-silent")
    record("(생활비 조회 후) 저축으로 3만 원 -> 출금 = 생활비", state, answer,
           state.get("params", {}).get("from_account") == "생활비")

    ask("생활비랑 저축 잔액 보여줘", "switch-many")
    state, answer = ask("여행 자금으로 3만 원 보내줘", "switch-many")
    record("(두 계좌 조회 후) 여행 자금으로 3만 원 -> 출금 비움", state, answer,
           "from_account" not in state.get("params", {}))

    # ③ 보여준 후보를 순서로 고르기 — 되묻기(ask_more)에 답하는 길 (Step 8 루프 1)
    state, answer = ask("주식 계좌 잔액 알려줘", "candidates")
    candidates = (state.get("missing") or {}).get("candidates", [])
    state, answer = resume("두 번째 거", "candidates")
    record("(후보를 본 뒤) 두 번째 거", state, answer, len(candidates) > 1 and accounts_of(state) == {candidates[1]})

    # ④ 개별 질문 — 질문마다 새 대화. 앞 질문의 영향을 받지 않게
    cases = [
        ("생활비 잔액 얼마야?",                                  # 글자 그대로 → 해석 안내 없이 바로 답
         lambda s, a: a == "생활비 계좌 잔액은 520,000원입니다."),
        ("여행 갈 때 쓰는 통장 잔액 알려줘",                      # 뜻으로만 찾을 수 있음 → 찾되 해석을 밝힘
         lambda s, a: "'여행 자금' 계좌로 이해했어요." in a and "310,000원" in a),
        ("오늘 날씨 어때?", lambda s, a: s.get("intent") == UNSUPPORTED),
        ("휴가비 계좌 잔액 알려줘",                                # 없는 이름 → 되묻거나, 추측했다면 해석을 밝힘
         lambda s, a: a.startswith("말씀하신 계좌를 찾지 못했어요") or "이해했어요" in a),
        ("주식 계좌 잔액 알려줘", lambda s, a: a.startswith("말씀하신 계좌를 찾지 못했어요")),
    ]
    for number, (question, check) in enumerate(cases):
        state, answer = ask(question, f"single-{number}")
        record(question, state, answer, check(state, answer))

    # ⑤ Step 5 — 즉시이체 + 승인 흐름
    data_store.reset()
    before = data_store.WORKING_PATH.read_bytes()
    state, prompt = ask("생활비에서 저축으로 10만 원 보내줘", "t-approve")
    record("이체 요청 -> 승인 화면에서 멈춤", state, prompt, "100,000원을 즉시이체" in prompt and "420,000원" in prompt)
    results.append(("승인 전에는 장부가 그대로", data_store.WORKING_PATH.read_bytes() == before))
    state, answer = resume("예", "t-approve")
    record("'예' -> 실행", state, answer, balance("생활비") == 420000 and balance("저축") == 1950000)
    results.append(("실행 후 무결성 10개 규칙 통과", not any(check_integrity(data_store.load()).values())))

    before = ledger()
    ask("생활비에서 저축으로 10만 원 보내줘", "t-reject")
    state, answer = resume("취소", "t-reject")
    record("'취소' -> 잔액·거래 그대로", state, answer, "취소했어요" in answer and ledger() == before)

    ask("생활비에서 저축으로 10만 원 보내줘", "t-unclear")
    state, prompt = resume("음 잠깐만", "t-unclear")
    record("예/취소가 아닌데 바뀐 게 없음 -> 승인 화면을 다시", state, prompt,
           prompt.startswith("'예' 또는 '취소'로 답해 주세요."))
    state, answer = resume("취소", "t-unclear")
    results.append(("다시 보여준 뒤 '취소' -> 끝남", "취소했어요" in answer))

    before = ledger()
    ask("생활비에서 저축으로 10만 원 보내줘", "t-expired")
    g._clock_offset = g.APPROVAL_TTL + timedelta(minutes=1)      # 6분 뒤에 "예"를 누른 것처럼 (시계 주입)
    state, answer = resume("예", "t-expired")
    g._clock_offset = timedelta(0)
    record("5분 지난 '예' -> 실행 안 함", state, answer, "승인 시간" in answer and ledger() == before)

    state, answer = ask("비상금에서 저축으로 100만 원 보내줘", "t-insufficient")
    record("잔액 부족 -> 승인 화면 없이 거절 사유", state, answer,
           not waiting_prompt(state) and "잔액이 부족해요" in answer)

    ask("생활비 잔액 얼마야?", "t-carried")
    state, prompt = ask("저축으로 3만 원 보내줘", "t-carried")
    record("(생활비 조회 후) 저축으로 3만 원 -> 이어받은 계좌를 밝힘", state, prompt,
           prompt.startswith("출금 계좌는 방금 보신 '생활비'"))
    resume("취소", "t-carried")

    ask("비상금에서 저축으로 7만 원 보내줘", "t-changed")      # 비상금 75,000원
    data = data_store.load()                                    # 승인을 기다리는 사이 다른 곳에서 1만 원 출금
    data["transactions"].append({
        "transaction_id": _next_id(data["transactions"], "transaction_id", "tx_"), "owner_id": "user_01",
        "account_id": "acc_004", "type": "withdrawal", "amount": 10000,
        "occurred_at": g._now().isoformat(timespec="seconds"), "card_id": None, "merchant": "확인용",
        "transfer_id": None,
    })
    next(a for a in data["accounts"] if a["account_id"] == "acc_004")["balance"] -= 10000
    data_store.save(data)
    state, answer = resume("예", "t-changed")
    record("승인 사이 잔액 감소 -> 실행 직전 재검사로 거절", state, answer,
           "승인하시는 사이" in answer and balance("비상금") == 65000)

    # ⑥ Step 6 — 처리 기록 (잔액 부족은 승인 화면 전이라 기록 없음)
    requests = data_store.load()["requests"]
    statuses = [r["status"] for r in requests]
    results.append((f"처리 기록: {statuses}",
                    statuses == ["completed", "cancelled", "cancelled", "expired", "cancelled", "failed"]))
    results.append(("완료 기록에 transfer_id tr_001, 요청·완료 시각",
                    requests[0]["details"].get("transfer_id") == "tr_001"
                    and requests[0]["requested_at"] <= requests[0]["finished_at"]))
    results.append(("재검사 실패 기록에 사유", "잔액이 부족해요" in (requests[-1]["reason"] or "")))

    real_replace, calls = data_store.os.replace, []            # 저장 실패 — 파일 교체가 재시도 횟수만큼 모두 실패

    def locked_replace(src, dst):
        calls.append(1)
        if len(calls) <= data_store.SAVE_ATTEMPTS:              # apply_change의 저장만 실패시키고, 기록 저장은 통과
            raise PermissionError("잠긴 파일 흉내")
        real_replace(src, dst)

    ask("생활비에서 저축으로 1만 원 보내줘", "t-savefail")
    before_balance = balance("생활비")
    data_store.os.replace = locked_replace
    try:
        state, answer = resume("예", "t-savefail")
    finally:
        data_store.os.replace = real_replace
    last = data_store.load()["requests"][-1]
    record("저장 실패(3번 모두) -> 이체 안 됨, 실패 기록", state, answer,
           "저장하지 못해" in answer and balance("생활비") == before_balance
           and last["status"] == "failed" and "저장하지 못해" in last["reason"])

    # ⑦ Step 7 — 카드 일시 잠금 (graph.py에 카드 전용 코드 없이)
    state, prompt = ask("생활비 카드 잠가줘", "t-card")
    record("카드 잠금 요청 -> 승인 화면", state, prompt, prompt.startswith("생활비 카드를 일시 잠금할게요."))
    state, answer = resume("예", "t-card")
    last = data_store.load()["requests"][-1]
    record("'예' -> locked + 완료 기록(card_id)", state, answer,
           card_status("생활비 카드") == "locked" and last["status"] == "completed"
           and last["details"] == {"card_id": "card_001"})
    state, answer = ask("생활비 카드 잠가줘", "t-card-again")
    record("이미 잠긴 카드 -> 승인 없이 안내", state, answer, not waiting_prompt(state) and "이미 잠겨 있어요" in answer)
    state, answer = ask("주식 카드 잠가줘", "t-card-unknown")
    record("없는 카드 -> 카드 후보로 안내", state, answer,
           answer.startswith("말씀하신 카드를 찾지 못했어요") and "여행 카드" in answer and "비상금" not in answer)

    # ⑧ Step 8 — 되묻기(루프 1)와 승인 전 수정(루프 2)
    state, answer = ask("저축으로 10만 원 옮겨줘", "t-ask")
    record("출금 계좌 없음 -> 되묻기 + 후보", state, answer,
           answer.startswith("출금 계좌를 알려 주세요.") and "생활비" in answer)
    state, answer = resume("생활비에서", "t-ask")
    record("'생활비에서' -> 이전 값(저축, 10만 원) 유지하고 승인 화면", state, answer,
           "생활비 → 저축, 100,000원을 즉시이체" in answer)
    resume("취소", "t-ask")

    before_balance = balance("생활비")
    ask("생활비에서 저축으로 10만 원 보내줘", "t-revise")
    state, answer = resume("아니, 5만 원만", "t-revise")
    record("승인 화면에서 '아니, 5만 원만' -> 다시 검사한 승인 화면", state, answer, "50,000원을 즉시이체" in answer)
    state, answer = resume("생활비 잔액 얼마야?", "t-revise")
    record("승인 대기 중 다른 업무 -> 지금 요청 유지", state, answer,
           answer.startswith("'예' 또는 '취소'로 답해 주세요.") and "50,000원" in answer)
    state, answer = resume("아니 1000만 원으로", "t-revise")
    record("잔액 넘는 금액으로 수정 -> 사유 + 이전 내용(5만 원)으로 다시", state, answer,
           "잔액이 부족해요" in answer and "50,000원을 즉시이체" in answer)
    state, answer = resume("예", "t-revise")
    record("'예' -> 이전 내용(5만 원)으로 실행", state, answer, balance("생활비") == before_balance - 50000)

    state, answer = ask("카드 잠가줘", "t-card-ask")
    record("어느 카드인지 없음 -> 카드 후보로 되묻기", state, answer,
           answer.startswith("카드를 알려 주세요.") and "고를 수 있는 카드: 생활비 카드" in answer)
    state, answer = resume("저축 카드", "t-card-ask")
    record("'저축 카드' -> 승인 화면", state, answer, answer.startswith("저축 카드를 일시 잠금할게요."))
    resume("취소", "t-card-ask")

    count = len(data_store.load()["requests"])
    ask("여행 자금으로 보내줘", "t-ask-stop")
    state, answer = resume("취소", "t-ask-stop")
    record("되묻기 중 '취소' -> 그만두고 기록 없음", state, answer,
           "그만둘게요" in answer and len(data_store.load()["requests"]) == count)

    # ⑨ 가짜 LLM — 진짜 LLM으로는 일부러 만들 수 없는 경우 (API 호출 없음, 2026-09-28 검수에서 추가)
    real_llm = g._get_llm
    transfer = {"intent": "transfer_instant", "from_account": "생활비", "to_account": "저축", "amount": 10000}
    try:
        # 수정 요청을 해석하다 LLM이 실패해도, 승인을 기다리던 요청을 잃지 않는다
        g._get_llm = lambda: fake
        fake = ScriptedLLM([transfer, RuntimeError("네트워크 오류 흉내")])
        ask("생활비에서 저축으로 1만 원", "fake-fail")
        state, prompt = resume("음...", "fake-fail")
        record("수정 해석 중 LLM 실패 -> 승인 화면 유지", state, prompt,
               prompt.startswith("'예' 또는 '취소'로 답해 주세요.") and "10,000원을 즉시이체" in prompt)
        resume("취소", "fake-fail")

        # 승인 화면 → 수정 → 되묻기 → "취소"여도, 승인 화면까지 간 요청은 기록한다
        fake = ScriptedLLM([transfer, {**transfer, "from_account": "uncertain"}])
        count = len(data_store.load()["requests"])
        ask("생활비에서 저축으로 1만 원", "fake-stop")
        resume("출금 계좌 바꿀래", "fake-stop")
        state, answer = resume("취소", "fake-stop")
        requests = data_store.load()["requests"]
        record("승인 후 수정 중 되묻기에서 '취소' -> cancelled 기록", state, answer,
               len(requests) == count + 1 and requests[-1]["status"] == "cancelled")
    finally:
        g._get_llm = real_llm

    results.append(("모든 시나리오 뒤 무결성 10개 규칙 통과", not any(check_integrity(data_store.load()).values())))
    data_store.reset()

    for description, passed in results:
        print(f"[{'OK  ' if passed else 'FAIL'}] {description}")
    print(f"결과: {sum(1 for _, passed in results if passed)}/{len(results)} 통과")


if __name__ == "__main__":
    run()
