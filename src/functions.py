"""업무별 조회·변경 로직을 담당한다.

그래프(graph.py)는 "절차"만 알고, "무슨 업무인지"는 여기 Handler들이 안다.
업무를 추가할 때 그래프를 고치지 않고 이 파일(과 intents.py)에만 추가하는 것이 목표.

각 Handler가 반드시 갖춰야 할 3가지:
    validate(params, data) -> (plan, error)   실행 가능한지 검사하고 처리안을 만든다
    describe(plan)         -> str             승인 화면에 보여줄 문장을 만든다
    apply(plan, data)      -> result          실제로 데이터를 바꾼다

필요한 slot 목록은 Handler가 아니라 intents.py에 있다.
(같은 정보를 두 곳에 적으면 결국 어긋나므로 한 곳에만 둔다)
"""

# TODO: CURRENT_USER_ID = "user_01"
#       인증(로그인)은 범위 밖. 모든 조회·변경은 이 사용자의 데이터만 대상으로 한다 (인가)

# TODO: 조회 함수 (승인 불필요)
#       - account_list_with_balance(data, accounts=None)
#         accounts가 비어 있으면 내 계좌 전체, 있으면 그 이름의 계좌들만
#         이름은 understand에서 이미 "내 계좌 이름 목록" 중에서 골라진 값 (검색이 아니라 걸러내기만 함)

# TODO: 변경 Handler (승인 필요) — 클래스 이름은 intent 이름을 PascalCase로 바꾼 것
#       - TransferInstantHandler      intent "transfer_instant"
#       - CardLockTemporaryHandler    intent "card_lock_temporary"

# TODO: HANDLERS — intent 이름 → Handler 인스턴스
#       키 목록이 intents.py의 WRITE_INTENTS와 정확히 같아야 한다
