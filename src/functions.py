"""업무별 조회·변경 로직을 담당한다.

그래프(graph.py)는 "절차"만 알고, "무슨 업무인지"는 여기 함수와 Handler들이 안다.
업무를 추가할 때 그래프를 고치지 않고 이 파일(과 intents.py)에만 추가하는 것이 목표.

각 Handler가 반드시 갖춰야 할 3가지 (Step 5부터):
    validate(params, data) -> (plan, error)   업무 검사 후 처리안을 만든다. 거절은 에러가 아니라 사유로 돌려준다
    describe(plan)         -> str             승인 화면에 보여줄 문장을 만든다
    apply(plan, data)      -> result          실제로 데이터를 바꾼다

필요한 slot 목록은 Handler가 아니라 intents.py에 있다.
(같은 정보를 두 곳에 적으면 결국 어긋나므로 한 곳에만 둔다)

단독 실행:  uv run python src/functions.py   (Step 2·3 완료 기준 자체 확인, API 호출 없음)
"""

CURRENT_USER_ID = "user_01"   # 인증(로그인)은 범위 밖. 모든 조회·변경은 이 사용자의 데이터만 대상으로 한다 (인가)


# ── 조회 (승인 불필요, 데이터를 읽기만 한다) ─────────────────────
# 이 파일은 data_store.save를 import하지 않는다 → 조회가 데이터를 바꿀 방법 자체가 없다 (설계 원칙 2)

def _my_accounts(data: dict) -> list[dict]:
    """현재 사용자의 계좌만 돌려준다 (인가). 다른 소유자의 계좌는 여기서 걸러진다."""
    return [acc for acc in data["accounts"] if acc["owner_id"] == CURRENT_USER_ID]


def _my_cards(data: dict) -> list[dict]:
    """현재 사용자의 카드만 돌려준다 (인가)."""
    return [card for card in data["cards"] if card["owner_id"] == CURRENT_USER_ID]


def my_account_names(data: dict) -> list[str]:
    """내 계좌 이름 목록. understand가 LLM에게 줄 허용 목록을 만들 때 쓴다 (이름만, 잔액은 없음)."""
    return [acc["nickname"] for acc in _my_accounts(data)]


def my_card_names(data: dict) -> list[str]:
    """내 카드 이름 목록. 허용 목록용."""
    return [card["name"] for card in _my_cards(data)]


def account_list_with_balance(data: dict, accounts: list[str] | None = None) -> list[dict]:
    """계좌 목록과 잔액을 조회한다.

    accounts: 조회할 계좌 이름 목록. 비어 있거나 None이면 내 계좌 전체.
              understand가 "내 계좌 이름 목록" 중에서 골라 넘긴 값이다 (검색하지 않고 걸러내기만 한다).

    돌려주는 값: [{"account_id", "nickname", "balance"}, ...]  — 데이터에 저장된 순서
        account_id  사용자에게 보여주지 않는다. "그 계좌에서 보내줘" 같은 후속 대화에 이어 쓰기 위한 것
        balance     숫자 그대로 (520000). "520,000원"처럼 꾸미는 건 respond의 일

    내 계좌에 없는 이름이 들어오면 ValueError — 앞 단계가 약속(허용 목록 안의 이름)을 어긴 버그이므로
    조용히 빼지 않고 드러낸다 (docs/설계서.md 설계 원칙 6).
    """
    mine = _my_accounts(data)

    if accounts:                                              # 이름이 지정된 경우에만 걸러낸다
        my_names = {acc["nickname"] for acc in mine}
        unknown = [name for name in accounts if name not in my_names]
        if unknown:
            raise ValueError(f"내 계좌에 없는 이름이 전달됐습니다: {unknown}")
        wanted = set(accounts)
        mine = [acc for acc in mine if acc["nickname"] in wanted]

    # 필요한 칸만 새 dict로 만들어 돌려준다
    # → owner_id 같은 내부 정보가 빠지고, 받는 쪽이 결과를 고쳐도 원래 데이터는 바뀌지 않는다
    return [
        {"account_id": acc["account_id"], "nickname": acc["nickname"], "balance": acc["balance"]}
        for acc in mine
    ]


def format_account_list_with_balance(rows: list[dict]) -> str:
    """조회 결과를 사용자에게 보여줄 문장으로 만든다 (정해진 틀, LLM 사용 안 함).

    숫자는 코드가 그대로 옮기므로 자릿수가 틀릴 일이 없다. account_id는 보여주지 않는다.
    """
    if len(rows) == 1:
        row = rows[0]
        return f"{row['nickname']} 계좌 잔액은 {row['balance']:,}원입니다."
    lines = [f"- {row['nickname']}: {row['balance']:,}원" for row in rows]
    return "계좌 잔액입니다.\n" + "\n".join(lines)


# 조회 업무 표 — intent 이름 → {실행 함수, 문장 틀}
# graph.py의 read_task와 respond는 이 표에서 찾아 쓰기만 한다 → 조회 업무가 늘어도 그래프는 그대로
# 키 목록은 intents.py의 READ_INTENTS와 정확히 같아야 한다 (graph.build_graph가 시작할 때 확인)
READ_TASKS = {
    "account_list_with_balance": {
        "run": account_list_with_balance,
        "format": format_account_list_with_balance,
    },
}


# ── 변경 (승인 필요) — Step 5, 7에서 만든다 ─────────────────────
# TODO: TransferInstantHandler      intent "transfer_instant"
# TODO: CardLockTemporaryHandler    intent "card_lock_temporary"
# TODO: HANDLERS — intent 이름 → Handler 인스턴스. 키 목록이 intents.py의 WRITE_INTENTS와 정확히 같아야 한다


# ── 단독 실행: Step 2 완료 기준 확인 ─────────────────────────────
def _self_check() -> None:
    """구현계획 Step 2·3의 완료 기준을 확인한다. 작업용 복사본을 읽기만 하고 바꾸지 않는다."""
    import copy
    from data_store import load                               # 확인할 때만 필요해서 여기서 불러온다. save는 불러오지 않는다

    data = load()
    before = copy.deepcopy(data)                              # 조회 전 상태를 통째로 떠둠 (나중에 안 바뀌었는지 비교)
    results = []                                              # (확인 내용, 통과 여부) 목록

    everything = account_list_with_balance(data)
    results.append(("지정하지 않으면 내 계좌 4개", len(everything) == 4))
    results.append(("다른 소유자(user_02)의 계좌는 없음", all(a["account_id"] != "acc_005" for a in everything)))
    results.append(("빈 목록을 줘도 전체 4개", len(account_list_with_balance(data, [])) == 4))

    one = account_list_with_balance(data, ["생활비"])
    results.append(("['생활비'] -> 1개, 내 생활비(acc_001)", len(one) == 1 and one[0]["account_id"] == "acc_001"))
    results.append(("['생활비', '저축'] -> 2개", len(account_list_with_balance(data, ["생활비", "저축"])) == 2))

    results.append(("돌려주는 칸은 account_id·nickname·balance뿐",
                    all(set(a) == {"account_id", "nickname", "balance"} for a in everything)))
    results.append(("잔액은 숫자 그대로", all(isinstance(a["balance"], int) for a in everything)))

    try:
        account_list_with_balance(data, ["휴가비"])           # 내 계좌에 없는 이름
        raised = False
    except ValueError:
        raised = True
    results.append(("내 계좌에 없는 이름이면 ValueError", raised))

    everything[0]["balance"] = 0                              # 받은 결과를 일부러 고쳐봐도
    results.append(("조회가 데이터를 바꾸지 않음", data == before))

    # Step 3에서 추가한 것
    from intents import READ_INTENTS
    results.append(("내 계좌 이름 4개, 내 카드 이름 3개 (user_02 제외)",
                    len(my_account_names(data)) == 4 and len(my_card_names(data)) == 3))
    results.append(("문장 틀: 1개면 한 문장, 금액에 쉼표",
                    format_account_list_with_balance(one) == "생활비 계좌 잔액은 520,000원입니다."))
    results.append(("READ_TASKS 키 = intents.py의 READ_INTENTS", set(READ_TASKS) == READ_INTENTS))

    for description, passed in results:
        print(f"[{'OK  ' if passed else 'FAIL'}] {description}")
    passed_count = sum(1 for _, passed in results if passed)
    print(f"결과: {passed_count}/{len(results)} 통과")


if __name__ == "__main__":   # 직접 실행했을 때만 확인 코드를 돌린다
    _self_check()
