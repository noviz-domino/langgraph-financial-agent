"""intent(사용자 의도) 목록의 유일한 정의처.

intent 이름을 쓰는 곳(understand 프롬프트, Pydantic 스키마, route_request, HANDLERS)은
전부 이 파일을 가져다 쓴다. 이름을 추가하거나 바꿀 때는 여기만 고친다.

이름 규칙 — 대상_동작_세부
    card_lock_temporary = card(카드) · lock(잠금) · temporary(일시적인)
    → 알파벳순으로 정렬하면 같은 대상(account, card, transfer ...)끼리 모인다

나누는 기준
    처리 방식이 다르면 intent를 나누고, 대상만 다르면 slot으로 구분한다.
    예) "계좌 목록"과 "생활비 잔액"은 같은 처리 → 하나의 intent + accounts slot
        "총 잔액"은 더하는 처리가 추가됨 → 별도 intent (2차)

slot(값을 담는 빈칸) 규칙 — 2026-09-26 수정
    required = 없으면 실행할 수 없음 → 사용자에게 되묻는다 (그래프의 ask_more)
    optional = 없어도 실행 가능 → 있으면 결과를 좁히는 데 쓴다

    계좌·카드를 가리키는 slot에는 LLM이 "내 계좌(카드) 이름 목록 + uncertain" 중에서
    고른 값이 들어간다. 사용자가 "여행 갈 때 쓰는 통장"이라고 해도 "여행 자금"이 들어간다.
        실제 이름     → 확실히 그 대상
        "uncertain"   → 말은 했는데 어느 것인지 모름 → 후보를 보여주며 되묻기
        비어 있음      → 아예 말하지 않음 → 필수 slot이면 되묻기

    역할 나누기(어느 게 출금이고 어느 게 입금인지)도 LLM이 slot별로 나눠 담는다.

검증은 두 층
    형식 — "LLM이 약속을 지켰나" (허용 목록 안의 값인가, 정수인가)
           → Pydantic 모델에 선언만 한다. 어기면 ValidationError (LLM의 실수)
    업무 — "사용자 요청이 실행할 수 있는 내용인가" (금액 > 0, 출금 ≠ 입금, 잔액)
           → Handler의 validate가 사유를 담아 거절한다 (업무상 거절, 에러 아님)
"""

INTENTS = {
    # ── 조회 (승인 불필요) ─────────────────────────────────────
    "account_list_with_balance": {
        "label": "계좌 조회",            # 사용자에게 보여줄 짧은 이름 (안내 문장에 쓴다)
        "desc": "계좌 목록과 잔액 조회. 계좌를 지정하면 그 계좌들만 보여준다",
        "kind": "read",
        "required": [],
        "optional": ["accounts"],        # 목록 — "생활비랑 저축 잔액"처럼 여러 개도 담을 수 있음
    },
    # ── 변경 (승인 필요) ───────────────────────────────────────
    "transfer_instant": {
        "label": "즉시이체",
        "desc": "즉시이체. 내 계좌 사이에서 지정한 금액을 옮긴다",
        "kind": "write",
        "required": ["from_account", "to_account", "amount"],
        "optional": [],
    },
    "card_lock_temporary": {
        "label": "카드 일시 잠금",
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

# ── slot 표 ─────────────────────────────────────────────────────
# 칸마다 "무엇을 가리키는 칸인가"를 여기에만 적는다 (2026-09-28, Step 7 — 전에는 graph.py에 흩어져 있었음).
#   kind    이름을 고르는 칸이면 그 종류 (account / card). 후보 목록은 functions.NAME_LISTS[kind]에서 찾는다
#   role    사용자에게 칸을 가리킬 때 쓰는 이름 ("출금 계좌를 알려 주세요")
#   suffix  고른 이름 뒤에 붙일 말 ("'여행 자금' 계좌로 이해했어요"). 카드 이름은 이미 "…카드"라 빈칸
SLOTS = {
    "accounts":     {"kind": "account", "role": "조회 계좌", "suffix": "계좌"},
    "from_account": {"kind": "account", "role": "출금 계좌", "suffix": "계좌"},
    "to_account":   {"kind": "account", "role": "입금 계좌", "suffix": "계좌"},
    "amount":       {"kind": None,      "role": "금액",      "suffix": ""},
    "card":         {"kind": "card",    "role": "카드",      "suffix": ""},
}
KIND_LABELS = {"account": "계좌", "card": "카드"}    # "말씀하신 계좌(카드)를 찾지 못했어요"
NAME_SLOTS = [slot for slot, info in SLOTS.items() if info["kind"]]   # 이름을 고르는 칸들 (자동 생성)

# ── INTENTS 표 밖의 특별한 값 ─────────────────────────────────────
UNSUPPORTED = "unsupported"          # intent — 이 에이전트가 처리할 수 없는 요청 (LLM이 고른다)
NOT_UNDERSTOOD = "not_understood"    # intent — LLM이 형식을 어겼거나 알 수 없는 이유로 실패했다 (코드가 넣는다)
MULTIPLE_REQUESTS = "multiple_requests"   # intent — 한 번에 업무를 둘 이상 요청했다 (LLM이 표시, 코드가 넣는다)
LLM_BUSY = "llm_busy"                # intent — AI 서버가 바쁘거나(503·429) 시간 안에 답하지 않았다 (코드가 넣는다)
UNCERTAIN = "uncertain"              # slot — 계좌·카드를 말했지만 목록 중 어느 것인지 모른다 (LLM이 고른다)
