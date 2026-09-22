"""JSON 데이터의 복사·읽기·저장을 담당한다.

원본(initial_data.json)은 절대 수정하지 않고, 작업용 복사본(bank_data.json)만 다룬다.
다른 모듈은 파일 경로나 json 모듈을 직접 만지지 않고 반드시 이 모듈을 거친다.
(저장하는 곳을 한 군데로 모아야 문제가 생겼을 때 추적이 쉽다)
"""

# TODO: ensure_working_copy()  — 작업용 복사본이 없으면 원본을 복사해서 만든다
# TODO: load()                 — 작업용 복사본을 읽어 dict로 돌려준다
# TODO: save(data)             — dict를 작업용 복사본에 쓴다. 쓰기 도중 중단돼도
#                                파일이 깨지지 않도록 임시 파일에 쓴 뒤 교체한다(atomic write)
# TODO: reset()                — 원본을 다시 복사해 초기 상태로 되돌린다 (테스트용)
