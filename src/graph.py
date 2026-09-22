"""State 정의, 노드, 조건부 Edge, 승인 흐름을 담당한다.

노드 9개 / 조건부 Edge 4개 / 루프 2개 / 중단 지점 2개로 구성.
자세한 설계 근거는 devlog/2026-09-22_워크플로우-설계-이해.md 참고.
"""

# TODO: AgentState (TypedDict)
#       messages / intent / params / missing / plan / decision / result / request_id / revision_count

# ── 노드 9개 ────────────────────────────────────────────
# TODO: understand      의도·파라미터 추출 후 기존 params에 "병합"(덮어쓰기 아님)
# TODO: ask_more        빠진 정보를 묻고 interrupt로 중단
# TODO: read_task       데이터를 읽기만 함 (save를 부르지 않는다)
# TODO: plan_change     처리안 작성 + 사전 검증. 아직 데이터를 바꾸지 않는다
# TODO: confirm_change  처리안을 보여주고 interrupt로 중단, 응답을 3갈래로 분류
# TODO: apply_change    재검증 후 실제 변경 + 저장 (TOCTOU 방지)
# TODO: rollback        저장 실패 시 변경 전 상태 유지를 보장하고 사유를 남김
# TODO: record_result   requests 배열에 성공·실패·취소를 기록
# TODO: respond         결과를 사람이 읽을 문장으로 생성

# ── 조건부 Edge 4개 (노드가 아니라 라우팅 함수) ──────────
# TODO: route_request     missing / intent를 보고 ask_more · read_task · plan_change 중 선택
# TODO: route_validation  plan이 만들어졌는지 보고 confirm_change · respond 중 선택
# TODO: route_decision    decision을 보고 apply_change · record_result · plan_change 중 선택
# TODO: route_save        result.ok를 보고 record_result · rollback 중 선택

# TODO: build_graph()  — StateGraph에 위 노드/Edge를 등록하고 Checkpointer와 함께 compile
