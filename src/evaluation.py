"""LLM(understand)이 말을 제대로 알아들었는지 재서 LangSmith Experiment로 남긴다.

흔들릴 수 있는 곳은 LLM을 쓰는 understand 하나뿐이다. 나머지 노드는 코드라 scenarios.py로 한 번 확인하면 된다.
그래서 여기서는 사례마다 세 가지를 채점한다 — 무슨 업무로(intent), 무슨 내용으로(params), 어디서 멈췄나.
같은 사례를 여러 번 돌리면(--repeat) "몇 번 중 몇 번"을 잴 수 있다 (한 번 돌리는 scenarios.py로는 알 수 없음).

  Dataset    GOLDEN_SET (질문 + 기대값). 내용이 같으면 같은 Dataset을 다시 쓴다 → 실험끼리 비교 가능
  target     그래프를 평소처럼 실행하고 결과만 뽑는다 (그래프 안에 평가 코드는 없다)
  evaluator  아래 채점 함수들. 이 컴퓨터에서 실행되고 점수만 LangSmith에 올라간다

사례마다 새 대화(thread_id)와 새 장부(use_ledger)를 쓴다 — 작업용 복사본(bank_data.json)은 건드리지 않는다.

단독 실행:
  uv run python src/evaluation.py              # LangSmith Experiment (Gemini 4회)
  uv run python src/evaluation.py --repeat 5   # 사례마다 5번 (Gemini 20회)
  uv run python src/evaluation.py --offline    # 채점 함수만 가짜 LLM으로 확인 (API 없음)
"""

import argparse
import hashlib
import json
import logging
import os
from uuid import uuid4

from langchain_core.messages import HumanMessage

import data_store
import graph as g
from config import load_env

# ── Golden set: 입력(inputs)과 기대값(reference) ─────────────────
# stopped_at: 멈춘 노드. None = 멈추지 않고 답까지 끝남
GOLDEN_SET = [
    {"case_id": "balance",                                       # 기본 조회 — 승인 없이 바로 답
     "inputs": {"question": "생활비 잔액 얼마야?"},
     "reference": {"intent": "account_list_with_balance",
                   "params": {"accounts": ["생활비"]},
                   "stopped_at": None}},
    {"case_id": "transfer",                                      # 쓰기 업무 — '10만 원'을 숫자로, 승인에서 멈춤
     "inputs": {"question": "생활비에서 저축으로 10만 원 보내줘"},
     "reference": {"intent": "transfer_instant",
                   "params": {"from_account": "생활비", "to_account": "저축", "amount": 100000},
                   "stopped_at": "confirm_change"}},
    {"case_id": "missing_amount",                                # 정보 부족 — 금액을 지어내지 않고 되묻기
     "inputs": {"question": "생활비에서 저축으로 돈 보내줘"},
     "reference": {"intent": "transfer_instant",
                   "params": {"from_account": "생활비", "to_account": "저축"},
                   "stopped_at": "ask_more"}},
    {"case_id": "no_from_account",                               # B25 — 말하지 않은 출금 계좌를 지어내지 않기
     "inputs": {"question": "저축으로 10만 원 옮겨줘"},
     "reference": {"intent": "transfer_instant",
                   "params": {"to_account": "저축", "amount": 100000},
                   "stopped_at": "ask_more"}},
]

GUESS_LOG = "짐작한 출금 계좌를 비움"                             # graph.understand가 LLM의 짐작을 지울 때 남기는 로그


# ── target: 그래프 실행 → 채점에 쓸 값만 ─────────────────────────
class _GuessCatcher(logging.Handler):
    """이번 실행에서 LLM이 출금 계좌를 지어냈는지 — 코드가 지우면서 남긴 로그로 안다."""

    def __init__(self):
        super().__init__(level=logging.INFO)
        self.guessed = None

    def emit(self, record):
        if GUESS_LOG in record.getMessage():
            self.guessed = record.args[0] if record.args else "?"


def target(inputs: dict, app=None) -> dict:
    app = app or _app()
    catcher = _GuessCatcher()
    graph_logger = logging.getLogger("graph")
    graph_logger.setLevel(logging.INFO)
    graph_logger.addHandler(catcher)
    ledger_id = uuid4().hex
    config = {"configurable": {"thread_id": ledger_id}}
    try:
        with data_store.use_ledger(ledger_id):                  # 원본에서 새 장부 — 잔액이 늘 같은 데서 시작
            state = app.invoke({"messages": [HumanMessage(content=inputs["question"])]}, config)
    finally:
        graph_logger.removeHandler(catcher)
        (data_store.LEDGER_DIR / f"{ledger_id}.json").unlink(missing_ok=True)
    stopped = app.get_state(config).next                        # interrupt로 멈췄으면 그 노드, 끝났으면 ()
    return {
        "intent": state.get("intent"),
        "params": {k: v for k, v in (state.get("params") or {}).items() if v is not None},
        "stopped_at": stopped[0] if stopped else None,
        "llm_guessed_from": catcher.guessed,
    }


_graph = None


def _app():
    global _graph
    if _graph is None:
        _graph = g.build_graph()
    return _graph


# ── evaluator: 채점 함수 (key = LangSmith에 보이는 점수 이름) ──────
def _compare(key: str, outputs: dict, reference_outputs: dict) -> dict:
    actual, expected = outputs[key], reference_outputs[key]
    return {"key": f"{key}_correct", "score": actual == expected, "comment": f"실제 {actual} / 기대 {expected}"}


def intent_correct(outputs: dict, reference_outputs: dict) -> dict:
    return _compare("intent", outputs, reference_outputs)


def params_correct(outputs: dict, reference_outputs: dict) -> dict:
    return _compare("params", outputs, reference_outputs)


def stopped_correct(outputs: dict, reference_outputs: dict) -> dict:
    return _compare("stopped_at", outputs, reference_outputs)


def llm_no_guess(outputs: dict) -> dict:
    """LLM 자체가 출금 계좌를 지어내지 않았나. 지어내도 코드가 지우므로 params_correct와 따로 본다 (안전장치 두 겹)."""
    guessed = outputs["llm_guessed_from"]
    return {"key": "llm_no_guess", "score": guessed is None,
            "comment": "LLM이 지어내지 않음" if guessed is None else f"LLM이 '{guessed}'를 지어냄 → 코드가 비움"}


EVALUATORS = [intent_correct, params_correct, stopped_correct, llm_no_guess]


# ── Experiment ───────────────────────────────────────────────────
def _dataset_name() -> str:
    """사례 내용으로 이름을 정한다 — 같으면 같은 Dataset(실험끼리 비교), 사례를 고치면 새 Dataset."""
    digest = hashlib.sha1(json.dumps(GOLDEN_SET, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    return f"financial-agent-understand-{digest[:8]}"


def run_experiment(repeat: int) -> None:
    load_env()                                                  # LangSmith 키를 Client()보다 먼저 읽는다
    if not os.getenv("LANGSMITH_API_KEY"):
        raise SystemExit("LANGSMITH_API_KEY가 없어요. .env에 넣거나 --offline으로 실행하세요.")
    from langsmith import Client

    client = Client()
    name = _dataset_name()
    if client.has_dataset(dataset_name=name):
        dataset = client.read_dataset(dataset_name=name)
    else:
        dataset = client.create_dataset(dataset_name=name, description="understand 채점용 Golden set")
        client.create_examples(dataset_id=dataset.id, examples=[
            {"inputs": c["inputs"], "outputs": c["reference"], "metadata": {"case_id": c["case_id"]}}
            for c in GOLDEN_SET
        ])

    experiment = client.evaluate(
        target, data=dataset.id, evaluators=EVALUATORS,
        experiment_prefix="understand", num_repetitions=repeat,
        max_concurrency=1,                                      # 무료 등급 — 한 번에 하나씩
        metadata={"model": g.MODEL_NAME, "repeat": repeat},
    )

    print(f"\nDataset: {name}  (사례 {len(GOLDEN_SET)}개 × {repeat}번)")
    totals: dict[str, list] = {}
    for row in experiment:
        case_id = row["example"].metadata["case_id"]
        scores = {r.key: r.score for r in row["evaluation_results"]["results"]}
        for key, score in scores.items():
            totals.setdefault(key, []).append(bool(score))
        marks = " ".join(f"{k}={'O' if v else 'X'}" for k, v in scores.items())
        print(f"  [{case_id}] {marks}" + (f"  실행 오류: {row['run'].error}" if row["run"].error else ""))
    for key, values in totals.items():
        print(f"  {key}: {sum(values)}/{len(values)}")


# ── 오프라인 확인: 채점 함수가 맞게 가르는지 (가짜 LLM, API 없음) ───
def _self_check_offline() -> None:
    from scenarios import ScriptedLLM                           # 정해진 답을 내는 가짜 LLM

    right = {                                                   # 사례마다 LLM이 올바로 답한 경우
        "balance": {"intent": "account_list_with_balance", "accounts": ["생활비"]},
        "transfer": {"intent": "transfer_instant", "from_account": "생활비", "to_account": "저축", "amount": 100000},
        "missing_amount": {"intent": "transfer_instant", "from_account": "생활비", "to_account": "저축"},
        "no_from_account": {"intent": "transfer_instant", "to_account": "저축", "amount": 100000},
    }
    def ledgers() -> set:
        return set(data_store.LEDGER_DIR.glob("*.json")) if data_store.LEDGER_DIR.exists() else set()

    checks = []
    before = ledgers()
    real_llm = g._get_llm
    try:
        for case in GOLDEN_SET:
            g._get_llm = lambda answer=right[case["case_id"]]: ScriptedLLM([answer])
            outputs = target(case["inputs"], g.build_graph())
            scores = [e(outputs, case["reference"])["score"] for e in EVALUATORS[:3]] + [llm_no_guess(outputs)["score"]]
            checks.append((f"{case['case_id']}: 올바른 답 → 네 항목 모두 통과", all(scores)))

        # B25 — LLM이 출금 계좌를 지어낸 경우: 코드가 지워서 결과는 맞지만, llm_no_guess는 잡아내야 한다
        case = next(c for c in GOLDEN_SET if c["case_id"] == "no_from_account")
        g._get_llm = lambda: ScriptedLLM([{**right["no_from_account"], "from_account": "생활비"}])
        outputs = target(case["inputs"], g.build_graph())
        checks.append(("지어낸 출금 계좌 → llm_no_guess 실패", llm_no_guess(outputs)["score"] is False))
        checks.append(("지어낸 출금 계좌 → 코드가 비워서 params·멈춘 곳은 통과",
                       params_correct(outputs, case["reference"])["score"]
                       and stopped_correct(outputs, case["reference"])["score"]))

        # 금액을 틀리게 알아들은 경우 → params만 실패
        case = next(c for c in GOLDEN_SET if c["case_id"] == "transfer")
        g._get_llm = lambda: ScriptedLLM([{**right["transfer"], "amount": 10000}])
        outputs = target(case["inputs"], g.build_graph())
        checks.append(("금액 1만 원으로 오해 → params 실패, intent 통과",
                       not params_correct(outputs, case["reference"])["score"]
                       and intent_correct(outputs, case["reference"])["score"]))
    finally:
        g._get_llm = real_llm

    checks.append(("평가용 장부가 남지 않음", ledgers() == before))

    for text, passed in checks:
        print(f"[{'PASS' if passed else 'FAIL'}] {text}")
    print(f"\n{sum(p for _, p in checks)}/{len(checks)} 통과")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="understand 평가 (LangSmith Experiment)")
    parser.add_argument("--repeat", type=int, default=1, help="사례마다 몇 번 돌릴지 (Gemini 호출 = 사례 수 × 이 값)")
    parser.add_argument("--offline", action="store_true", help="채점 함수만 가짜 LLM으로 확인 (API 없음)")
    args = parser.parse_args()
    if args.offline:
        _self_check_offline()
    else:
        run_experiment(args.repeat)
