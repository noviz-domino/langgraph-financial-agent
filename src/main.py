"""사용자 입력을 받아 그래프를 실행한다.

중단(interrupt)이 걸린 상태인지 확인해서, 새 요청인지 재개인지 구분하는 것이 핵심.
중단 종류(정보 보충 / 승인 요청)가 달라도 처리 코드는 한 벌이면 된다.
"""

# TODO: 그래프 생성 + thread_id 설정 (Checkpointer가 대화를 구분하는 단위)
# TODO: 입력 루프
#       - 중단 상태면  -> graph.invoke(Command(resume=사용자입력), config)
#       - 아니면       -> graph.invoke({"messages": [사용자입력]}, config)
# TODO: 재시작 시 진행 중이던 업무가 있으면 확인하고, decision은 비운 뒤 다시 승인받기
#       (명세: "이전 승인만으로 변경을 자동 실행하지 않는다")
