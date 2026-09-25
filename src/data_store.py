"""JSON 데이터의 복사·읽기·저장을 담당한다.

원본(initial_data.json)은 프로그램이 절대 수정하지 않고, 작업용 복사본(bank_data.json)만 다룬다.
다른 모듈은 파일 경로나 json 모듈을 직접 만지지 않고 반드시 이 모듈을 거친다.
(저장하는 곳을 한 군데로 모아야 문제가 생겼을 때 추적이 쉽고, 검사 관문도 하나만 세우면 된다)

결정 사항 (devlog/2026-09-26 참고)
    1. 저장 전에 무결성 검사를 하고, 실패하면 파일에 쓰지 않는다 (IntegrityError)
    2. 작업용 복사본이 없으면 load()가 원본에서 자동으로 만든다 (lazy initialization)
    3. atomic write — 같은 폴더의 임시 파일에 다 쓰고, 디스크에 기록한 뒤, 이름을 바꿔 교체한다
    4. UTF-8, 한글 그대로, 들여쓰기 2칸, 파일 끝 줄바꿈

단독 실행:  uv run python src/data_store.py   (Step 1 완료 기준 자체 확인)
"""

import json
import logging
import os
from pathlib import Path

from integrity import check_integrity   # 같은 src/ 폴더의 모듈 → 폴더 이름 없이 불러온다 (실행 방식 A안)

logger = logging.getLogger(__name__)    # 이 모듈 전용 기록기. 어떤 모듈이 남긴 기록인지 구분된다

# ── 경로 ─────────────────────────────────────────────────────────
# 이 파일(src/data_store.py) 위치 기준 → 어느 폴더에서 실행해도 같은 파일을 찾는다
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ORIGINAL_PATH = DATA_DIR / "initial_data.json"   # 원본 — 읽기만 한다
WORKING_PATH = DATA_DIR / "bank_data.json"       # 작업용 복사본 — 여기만 고친다
TEMP_PATH = DATA_DIR / "bank_data.json.tmp"      # atomic write용 임시 파일 (진짜 파일과 같은 폴더)


# ── 에러 ─────────────────────────────────────────────────────────
class DataStoreError(Exception):
    """data_store가 내는 모든 에러의 부모. 이걸로 받으면 아래 둘을 한꺼번에 받는다."""


class IntegrityError(DataStoreError):
    """무결성 규칙을 어겨서 저장을 거부했다. 파일은 저장 전 상태 그대로다. (재시도해도 소용없음)"""

    def __init__(self, violations: dict[str, list[str]]):
        self.violations = violations                         # 규칙별 위반 내용. 받는 쪽이 자세히 보고 싶을 때 쓴다
        count = sum(len(messages) for messages in violations.values())
        super().__init__(f"무결성 규칙 위반 {count}건으로 저장을 거부했습니다")   # 에러 메시지로 쓸 문장


class StorageError(DataStoreError):
    """파일을 읽거나 쓸 수 없다. (파일 잠김, 깨진 JSON, 원본 없음 등) (재시도하면 될 수도 있음)"""


# ── 내부 도우미 ──────────────────────────────────────────────────
def _read_json(path: Path) -> dict:
    """JSON 파일을 읽는다. 읽을 수 없으면 파이썬 기본 에러를 StorageError로 바꿔서 낸다."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise StorageError(f"파일이 없습니다: {path.name}") from e
    except json.JSONDecodeError as e:
        raise StorageError(f"JSON 형식이 깨졌습니다: {path.name} ({e.lineno}번째 줄)") from e
    except OSError as e:                                     # 권한 없음, 파일 잠김 등 나머지 파일 문제
        raise StorageError(f"파일을 읽을 수 없습니다: {path.name} ({type(e).__name__})") from e


# ── 공개 함수 ────────────────────────────────────────────────────
def save(data: dict) -> None:
    """작업용 복사본에 저장한다. 저장하는 길은 이 함수 하나뿐이다.

    [0] 무결성 검사 — 실패하면 IntegrityError (파일은 하나도 안 건드림)
    [1] 임시 파일에 끝까지 쓰고, 디스크에 확실히 기록
    [2] 임시 파일을 진짜 파일 이름으로 교체 — 실패하면 StorageError (진짜 파일은 옛날 그대로)
    """
    # [0] 무결성 검사
    violations = check_integrity(data)
    if any(violations.values()):                             # 규칙 중 하나라도 위반 목록이 비어 있지 않으면
        raise IntegrityError(violations)

    # 파일을 열기 "전에" 글자로 바꿔둔다 → 데이터에 버그가 있으면 여기서 터지고, 파일은 안 건드림
    text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"

    try:
        # [1] 임시 파일에 쓰기
        with open(TEMP_PATH, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()                                        # 파이썬이 들고 있던 내용을 운영체제에 넘김
            os.fsync(f.fileno())                             # 운영체제에게 "지금 당장 디스크에 써라"
        # [2] 이름 바꾸기 — 운영체제가 한 번에 처리 (옛 파일 아니면 새 파일, 반쪽은 없음)
        os.replace(TEMP_PATH, WORKING_PATH)
    except OSError as e:
        try:
            TEMP_PATH.unlink(missing_ok=True)                # 남은 임시 파일 정리 (없으면 그냥 넘어감)
        except OSError:
            pass                                             # 정리에 실패해도, 원래 에러를 알리는 게 더 중요하다
        raise StorageError(f"저장하지 못했습니다: {WORKING_PATH.name} ({type(e).__name__})") from e


def ensure_working_copy() -> bool:
    """작업용 복사본이 없으면 원본에서 만든다. 새로 만들었으면 True, 이미 있었으면 False."""
    if WORKING_PATH.exists():
        return False
    save(_read_json(ORIGINAL_PATH))                          # 복사도 무결성 검사 + atomic write를 똑같이 거친다
    logger.info("작업용 복사본이 없어서 원본에서 새로 만들었습니다: %s", WORKING_PATH.name)
    return True


def load() -> dict:
    """작업용 복사본을 읽어 dict로 돌려준다. 복사본이 없으면 먼저 만든다."""
    ensure_working_copy()
    return _read_json(WORKING_PATH)


def reset() -> None:
    """원본을 다시 복사해 처음 상태로 되돌린다 (테스트용). 복사본에 쌓인 변경은 모두 사라진다."""
    save(_read_json(ORIGINAL_PATH))
    logger.info("작업용 복사본을 원본 상태로 되돌렸습니다: %s", WORKING_PATH.name)


# ── 단독 실행: Step 1 완료 기준 확인 ─────────────────────────────
def _self_check() -> None:
    """구현계획 Step 1의 완료 기준을 차례로 확인한다.

    ⚠️ 작업용 복사본을 지우고 다시 만든다. 끝나면 원본 상태로 되돌려둔다.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    results = []                                             # (확인 내용, 통과 여부) 목록
    original_before = ORIGINAL_PATH.read_bytes()             # 원본이 안 바뀌는지 비교하려고 미리 떠둠

    # 1·2. 복사본이 없는 상태에서 load() → 자동으로 만들어지고 키 7개
    WORKING_PATH.unlink(missing_ok=True)
    data = load()
    results.append(("복사본이 없으면 load()가 자동으로 만든다", WORKING_PATH.exists()))
    results.append(("load()가 최상위 키 7개를 돌려준다", len(data) == 7))

    # 3. save() 후 다시 load() 하면 바뀐 값이 읽힌다
    data["accounts"][0]["nickname"] = "확인용"                 # 별명은 무결성 규칙과 무관 → 저장 통과
    save(data)
    results.append(("save() 후 load()에 바뀐 값이 반영된다", load()["accounts"][0]["nickname"] == "확인용"))

    # 4. 무결성 위반이면 저장을 거부하고, 파일은 그대로다
    before = WORKING_PATH.read_bytes()
    broken = load()
    broken["accounts"][0]["balance"] += 1                    # 잔액만 1원 바꿈 → 규칙 1(잔액 일치) 위반
    try:
        save(broken)
        refused = False
    except IntegrityError:
        refused = True
    results.append(("무결성 위반이면 IntegrityError로 거부한다", refused))
    results.append(("거부된 뒤 파일은 저장 전 그대로다", WORKING_PATH.read_bytes() == before))

    # 5. reset() 하면 원본 상태로 돌아간다
    reset()
    results.append(("reset() 하면 원본 상태로 돌아간다", load()["accounts"][0]["nickname"] == "생활비"))

    # 6·7. 원본은 한 번도 바뀌지 않았고, 임시 파일도 남지 않았다
    results.append(("원본 파일은 변경되지 않았다", ORIGINAL_PATH.read_bytes() == original_before))
    results.append(("임시 파일이 남지 않았다", not TEMP_PATH.exists()))

    for description, passed in results:
        print(f"[{'OK  ' if passed else 'FAIL'}] {description}")
    passed_count = sum(1 for _, passed in results if passed)
    print(f"결과: {passed_count}/{len(results)} 통과")


if __name__ == "__main__":   # 직접 실행했을 때만 확인 코드를 돌린다 (다른 파일에서 import 할 때는 안 돌림)
    _self_check()
