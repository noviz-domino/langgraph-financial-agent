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

from datetime import datetime

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


# 종류별 이름 목록 — intents.SLOTS의 kind로 찾는다. graph.py는 종류를 몰라도 후보를 보여줄 수 있다 (Step 7)
NAME_LISTS = {
    "account": my_account_names,
    "card": my_card_names,
}


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


def josa(word: str, pair: str) -> str:
    """받침에 맞는 조사를 고른다. pair는 "을/를", "이/가", "은/는", "으로/로".

    josa("저축", "으로/로") → "으로",  josa("생활비", "으로/로") → "로",  josa("비상금", "을/를") → "을"
    "으로/로"는 ㄹ받침이면 "로" ("여행 자금"은 ㅁ받침이라 "으로"). 한글이 아니면 받침 없음으로 본다.
    """
    with_final, without_final = pair.split("/")
    code = ord(word[-1]) - 0xAC00                             # 한글 음절 "가"부터의 순번
    final = code % 28 if 0 <= code < 11172 else 0             # 받침 번호 (0 = 받침 없음, 8 = ㄹ)
    if pair == "으로/로" and final == 8:
        return without_final
    return with_final if final else without_final


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


# ── 변경 (승인 필요) ─────────────────────────────────────────────
# 형식 검사(필수 slot이 있나, uncertain이 아닌가)는 그래프가 intents.py를 보고 공통으로 한다.
# Handler는 "업무" 검사만 한다. 거절은 에러가 아니라 사유 문장으로 돌려준다 (설계서 6장 규칙 5)

def _next_id(items: list[dict], field: str, prefix: str) -> str:
    """'tx_030'까지 있으면 'tx_031'. 번호는 3자리 이상으로 맞춘다."""
    numbers = [int(item[field].removeprefix(prefix)) for item in items if item.get(field)]
    return f"{prefix}{max(numbers, default=0) + 1:03d}"


def new_request(data: dict, intent: str, details: dict, status: str, reason: str | None,
                requested_at: str, finished_at: datetime) -> dict:
    """처리 기록 한 건을 만들어 data["requests"]에 붙이고 돌려준다 (저장은 부르는 쪽이).

    승인 화면까지 간 요청만 기록한다 (Step 6 결정 2). 어느 업무든 같은 틀이고 details만 업무마다 다르다.
    details에는 이름이 아니라 ID를 적는다 — 계좌 별명은 바뀔 수 있다 (결정 3)
    """
    request = {
        "request_id": _next_id(data["requests"], "request_id", "req_"),
        "owner_id": CURRENT_USER_ID,
        "intent": intent,
        "details": details,
        "status": status,                                     # completed / cancelled / expired / failed
        "reason": reason,
        "requested_at": requested_at,
        "finished_at": finished_at.isoformat(timespec="seconds"),
    }
    data["requests"].append(request)
    return request


class TransferInstantHandler:
    """즉시이체 — 내 계좌 사이에서 금액을 옮긴다. intent "transfer_instant"."""

    def validate(self, params: dict, data: dict) -> tuple[dict | None, str | None]:
        """업무 검사 후 처리안을 만든다. (처리안, None) 또는 (None, 거절 사유).

        처리안을 만들 때(plan_change)와 실행 직전(apply_change)에 **같은 함수**를 부른다
        → 두 검사가 어긋날 일이 없다 (Step 5 결정 5).
        """
        amount = params["amount"]
        if amount <= 0:
            return None, f"이체 금액은 0원보다 커야 해요. (말씀하신 금액: {amount:,}원)"
        if params["from_account"] == params["to_account"]:
            return None, "출금 계좌와 입금 계좌가 같아요. 다른 계좌를 골라 주세요."

        mine = {acc["nickname"]: acc for acc in _my_accounts(data)}
        source, target = mine[params["from_account"]], mine[params["to_account"]]   # 허용 목록 밖이면 KeyError (버그)
        if source["balance"] < amount:
            return None, (f"{source['nickname']} 계좌 잔액이 부족해요. "
                          f"(잔액 {source['balance']:,}원, 이체 금액 {amount:,}원)")

        return {
            "from_account_id": source["account_id"], "from_name": source["nickname"],
            "to_account_id": target["account_id"], "to_name": target["nickname"],
            "amount": amount,
            "balance_after": source["balance"] - amount,
        }, None

    def details(self, plan: dict) -> dict:
        """처리 기록(requests)에 남길 값 — ID와 금액만."""
        return {"from_account_id": plan["from_account_id"], "to_account_id": plan["to_account_id"],
                "amount": plan["amount"]}

    def describe(self, plan: dict) -> str:
        """승인 화면에 보여줄 문장 (정해진 틀, Step 5 결정 2). 숫자는 처리안의 값을 그대로 옮긴다."""
        return (f"{plan['from_name']} → {plan['to_name']}, {plan['amount']:,}원을 즉시이체할게요.\n"
                f"이체 후 {plan['from_name']} 잔액은 {plan['balance_after']:,}원이에요.")

    def apply(self, plan: dict, data: dict, now: datetime) -> tuple[str, dict]:
        """data를 실제로 바꾼다 (저장은 하지 않는다 — 부르는 쪽이 data_store.save).

        잔액 두 곳 + 거래 두 줄을 함께 바꾼다. 두 줄은 같은 transfer_id로 묶인다 (무결성 규칙 9).
        돌려주는 값: (완료 안내 문장, 처리 기록에 더할 값 {"transfer_id": ...})
        """
        accounts = {acc["account_id"]: acc for acc in data["accounts"]}
        accounts[plan["from_account_id"]]["balance"] -= plan["amount"]
        accounts[plan["to_account_id"]]["balance"] += plan["amount"]

        transfer_id = _next_id(data["transactions"], "transfer_id", "tr_")
        occurred_at = now.isoformat(timespec="seconds")
        for account_id, tx_type in [(plan["from_account_id"], "withdrawal"), (plan["to_account_id"], "deposit")]:
            data["transactions"].append({
                "transaction_id": _next_id(data["transactions"], "transaction_id", "tx_"),
                "owner_id": CURRENT_USER_ID, "account_id": account_id, "type": tx_type,
                "amount": plan["amount"], "occurred_at": occurred_at,
                "card_id": None, "merchant": None, "transfer_id": transfer_id,
            })

        balance = accounts[plan["from_account_id"]]["balance"]
        to_name = plan["to_name"]
        message = (f"{plan['from_name']}에서 {to_name}{josa(to_name, '으로/로')} {plan['amount']:,}원을 보냈어요.\n"
                   f"{plan['from_name']} 잔액은 {balance:,}원이에요.")
        return message, {"transfer_id": transfer_id}


class CardLockTemporaryHandler:
    """카드 일시 잠금 — 카드 상태를 active → locked로. intent "card_lock_temporary".

    계좌가 아닌 다른 데이터(cards)를 바꾸는 업무. 이 Handler를 추가하면서 graph.py를 고치지 않는지가
    Step 7의 검증 대상이다 (기획서 성공 기준 5).
    """

    def validate(self, params: dict, data: dict) -> tuple[dict | None, str | None]:
        """잠글 수 있는 카드인지 검사한다. 이미 잠겼거나 분실 정지된 카드는 거절 (Step 7 결정 1)."""
        card = {c["name"]: c for c in _my_cards(data)}[params["card"]]   # 허용 목록 밖이면 KeyError (버그)
        name = card["name"]
        if card["status"] == "locked":
            return None, f"{name}{josa(name, '은/는')} 이미 잠겨 있어요."
        if card["status"] == "reported_lost":
            return None, f"분실 정지된 카드는 잠글 수 없어요. {name} 재발급을 신청해 주세요."
        return {"card_id": card["card_id"], "card_name": name}, None

    def details(self, plan: dict) -> dict:
        return {"card_id": plan["card_id"]}

    def describe(self, plan: dict) -> str:
        name = plan["card_name"]
        return (f"{name}{josa(name, '을/를')} 일시 잠금할게요.\n"
                "잠그면 결제가 막히고, 분실 정지와 달리 나중에 풀 수 있어요.")

    def apply(self, plan: dict, data: dict, now: datetime) -> tuple[str, dict]:
        next(c for c in data["cards"] if c["card_id"] == plan["card_id"])["status"] = "locked"
        name = plan["card_name"]
        return f"{name}{josa(name, '을/를')} 잠갔어요.", {}


# 변경 업무 표 — intent 이름 → Handler. graph.py는 여기서 찾아 쓰기만 한다
HANDLERS = {
    "transfer_instant": TransferInstantHandler(),
    "card_lock_temporary": CardLockTemporaryHandler(),
}


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

    # Step 5에서 추가한 것 — 이체 Handler (파일에 저장하지 않고 복사본에서만 확인)
    from integrity import KST, check_integrity
    from intents import WRITE_INTENTS
    handler = HANDLERS["transfer_instant"]
    ok = {"from_account": "생활비", "to_account": "저축", "amount": 100000}

    plan, reason = handler.validate(ok, data)
    results.append(("이체 검사 통과 -> 처리안, 이체 후 잔액 420,000", reason is None and plan["balance_after"] == 420000))
    results.append(("승인 문장에 방향·금액·이체 후 잔액",
                    handler.describe(plan) == "생활비 → 저축, 100,000원을 즉시이체할게요.\n이체 후 생활비 잔액은 420,000원이에요."))
    results.append(("금액 0 이하 -> 거절 사유", handler.validate({**ok, "amount": -5000}, data)[1] is not None))
    results.append(("출금 = 입금 -> 거절 사유", handler.validate({**ok, "to_account": "생활비"}, data)[1] is not None))
    results.append(("잔액 부족 -> 거절 사유에 잔액", "75,000원" in (handler.validate(
        {"from_account": "비상금", "to_account": "저축", "amount": 100000}, data)[1] or "")))

    after = copy.deepcopy(data)
    now = datetime.now(KST)
    _, refs = handler.apply(plan, after, now)
    new_request(after, "transfer_instant", {**handler.details(plan), **refs}, "completed", None, now.isoformat(), now)
    balances = {a["nickname"]: a["balance"] for a in _my_accounts(after)}
    results.append(("apply 후 생활비 420,000 / 저축 1,950,000", balances["생활비"] == 420000 and balances["저축"] == 1950000))
    new_txs = after["transactions"][-2:]
    results.append(("거래 2줄이 같은 transfer_id", new_txs[0]["transfer_id"] == new_txs[1]["transfer_id"] == "tr_001"))
    results.append(("apply + 완료 기록 후 무결성 10개 규칙 통과", not any(check_integrity(after).values())))
    results.append(("완료 기록에 ID·금액·transfer_id, 번호 req_001",
                    after["requests"][0]["request_id"] == "req_001"
                    and after["requests"][0]["details"] == {"from_account_id": "acc_001", "to_account_id": "acc_002",
                                                            "amount": 100000, "transfer_id": "tr_001"}))
    no_record = copy.deepcopy(after)
    no_record["requests"].clear()                             # 이체만 저장하고 기록을 빠뜨린 버그를 흉내
    results.append(("완료 기록이 빠지면 규칙 10이 잡음", bool(check_integrity(no_record)["transfer_recorded"])))

    broken = copy.deepcopy(after)
    broken["transactions"].pop()                              # 입금 줄을 빠뜨린 버그를 흉내
    broken["accounts"][1]["balance"] -= 100000                # 잔액까지 맞춰도
    results.append(("입금 줄이 빠지면 규칙 9가 잡음", bool(check_integrity(broken)["transfer_pairs"])))

    # Step 7에서 추가한 것 — 카드 일시 잠금 Handler
    card = HANDLERS["card_lock_temporary"]
    card_plan, reason = card.validate({"card": "생활비 카드"}, data)
    results.append(("카드 잠금 검사 통과 -> 내 생활비 카드(card_001)", reason is None and card_plan["card_id"] == "card_001"))
    locked = copy.deepcopy(data)
    message, refs = card.apply(card_plan, locked, datetime.now(KST))
    results.append(("apply 후 status = locked, 안내 문장", locked["cards"][0]["status"] == "locked"
                    and message == "생활비 카드를 잠갔어요." and refs == {}))
    results.append(("이미 잠긴 카드 -> 거절 사유", card.validate({"card": "생활비 카드"}, locked)[1] == "생활비 카드는 이미 잠겨 있어요."))
    locked["cards"][0]["status"] = "reported_lost"
    results.append(("분실 정지 카드 -> 거절 사유", "분실 정지" in (card.validate({"card": "생활비 카드"}, locked)[1] or "")))
    locked["cards"][0]["status"] = "lock"                     # 잘못된 상태 값을 쓰는 버그를 흉내
    results.append(("잘못된 카드 상태는 무결성 규칙 6이 잡음", bool(check_integrity(locked)["values_valid"])))
    results.append(("NAME_LISTS: 카드 후보는 내 카드 3개", NAME_LISTS["card"](data) == ["생활비 카드", "저축 카드", "여행 카드"]))

    results.append(("HANDLERS 키 = WRITE_INTENTS (변경 업무 모두 Handler 있음)", set(HANDLERS) == WRITE_INTENTS))
    results.append(("조회·검사가 원래 데이터를 바꾸지 않음", data == before))

    for description, passed in results:
        print(f"[{'OK  ' if passed else 'FAIL'}] {description}")
    passed_count = sum(1 for _, passed in results if passed)
    print(f"결과: {passed_count}/{len(results)} 통과")


if __name__ == "__main__":   # 직접 실행했을 때만 확인 코드를 돌린다
    _self_check()
