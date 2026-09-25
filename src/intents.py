"""intent(사용자 의도) 목록의 유일한 정의처.

intent 이름을 쓰는 곳(understand 프롬프트, Pydantic 스키마, route_request, HANDLERS)은
전부 이 파일을 가져다 쓴다. 이름을 추가하거나 바꿀 때는 여기만 고친다.

이름 규칙 — 대상_동작_세부
    card_lock_temporary = card(카드) · lock(잠금) · temporary(일시적인)
    → 알파벳순으로 정렬하면 같은 대상(account, card, transfer ...)끼리 모인다

나누는 기준
    처리 방식이 다르면 intent를 나누고, 대상만 다르면 slot으로 구분한다.
    예) "계좌 목록"과 "생활비 잔액"은 같은 처리 → 하나의 intent + account slot
        "총 잔액"은 더하는 처리가 추가됨 → 별도 intent (2차)

slot(값을 담는 빈칸) 규칙
    required = 없으면 실행할 수 없음 → 사용자에게 되묻는다 (그래프의 ask_more)
    optional = 없어도 실행 가능 → 있으면 결과를 좁히는 데 쓴다
    slot에는 ID가 아니라 사용자가 말한 표현("생활비")이 들어간다.
    ID로 바꾸는 일은 read_task / plan_change가 한다 (후보가 여러 개면 되묻기 위해).
"""

INTENTS = {
    # ── 조회 (승인 불필요) ─────────────────────────────────────
    "account_list_with_balance": {
        "desc": "계좌 목록과 잔액 조회. 계좌를 지정하면 그 계좌만 보여준다",
        "kind": "read",
        "required": [],
        "optional": ["account"],
    },
    # ── 변경 (승인 필요) ───────────────────────────────────────
    "transfer_instant": {
        "desc": "즉시이체. 내 계좌 사이에서 지정한 금액을 옮긴다",
        "kind": "write",
        "required": ["from_account", "to_account", "amount"],
        "optional": [],
    },
    "card_lock_temporary": {
        "desc": "카드 일시 잠금. 분실 정지와 달리 잠금 해제로 되돌릴 수 있다",
        "kind": "write",
        "required": ["card"],
        "optional": [],
    },
}

# 아래 두 집합은 위 표에서 자동으로 만든다 → 손으로 따로 관리하지 않는다.
# frozenset = 한 번 만들면 바꿀 수 없는 집합. 실행 중에 실수로 추가·삭제되는 것을 막는다.
READ_INTENTS = frozenset(name for name, info in INTENTS.items() if info["kind"] == "read")
WRITE_INTENTS = frozenset(name for name, info in INTENTS.items() if info["kind"] == "write")
