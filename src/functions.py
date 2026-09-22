"""업무별 조회·변경 로직을 담당한다.

그래프(graph.py)는 "절차"만 알고, "무슨 업무인지"는 여기 Handler들이 안다.
업무를 추가할 때 그래프를 고치지 않고 이 파일에만 Handler를 추가하는 것이 목표.

각 Handler가 반드시 갖춰야 할 3가지:
    validate(params, data) -> (plan, error)   실행 가능한지 검사하고 처리안을 만든다
    describe(plan)         -> str             승인 화면에 보여줄 문장을 만든다
    apply(plan, data)      -> result          실제로 데이터를 바꾸고 저장한다
"""

# TODO: 조회 함수들 (승인 불필요)
#       - list_accounts, total_balance, list_transactions, list_cards, list_unpaid_bills

# TODO: 변경 업무 Handler들 (승인 필요)
#       - TransferHandler      즉시이체
#       - LockCardHandler      카드 일시 잠금
#       - PayBillHandler       청구서 납부

# TODO: HANDLERS  — intent 문자열로 담당 Handler를 찾는 레지스트리(등록부)
