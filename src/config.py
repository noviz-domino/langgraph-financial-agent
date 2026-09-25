"""환경 변수(API 키 등)를 불러온다.

불러오는 순서 — 먼저 불러온 값이 이긴다 (load_dotenv는 이미 있는 값을 덮어쓰지 않는다)
    1. 프로젝트 루트의 .env              이 프로젝트만의 설정. git에 올라가지 않는다
    2. SHARED_ENV_FILE이 가리키는 파일   여러 실습이 함께 쓰는 API 키 파일 (프로젝트 밖)

코드에 절대경로를 적지 않는 이유
    - 코드는 GitHub에 올라간다 → 내 컴퓨터의 사용자 이름과 폴더 구조가 공개된다
    - 다른 컴퓨터(채점자)에는 그 경로가 없다 → 채점자는 프로젝트 .env에 키를 직접 넣으면 된다

API 키 값은 절대 출력하거나 로그에 남기지 않는다.

단독 실행:  uv run python src/config.py   (키가 준비됐는지 True/False만 확인)
"""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ENV = Path(__file__).resolve().parent.parent / ".env"   # 프로젝트 루트의 .env
API_KEY_NAMES = ("GEMINI_API_KEY", "GOOGLE_API_KEY")            # Gemini 키 이름. 둘 중 하나만 있으면 된다


class ConfigError(Exception):
    """필요한 설정이 없어서 프로그램을 시작할 수 없다."""


def load_env() -> list[str]:
    """환경 변수를 불러오고, 실제로 읽은 파일의 종류를 돌려준다. API 키가 없으면 ConfigError.

    프로그램이 시작할 때 한 번 부른다 (main.py, 각 모듈의 확인 블록).
    """
    loaded = []

    if PROJECT_ENV.exists():
        load_dotenv(PROJECT_ENV)
        loaded.append("프로젝트 .env")

    shared = os.getenv("SHARED_ENV_FILE")                      # 프로젝트 .env에 적혀 있으면 여기서 읽힌다
    if shared:
        shared_path = Path(shared)
        if not shared_path.exists():                           # 경로를 적었는데 파일이 없으면 설정 실수
            raise ConfigError(f"SHARED_ENV_FILE이 가리키는 파일이 없습니다: {shared_path}")
        load_dotenv(shared_path)
        loaded.append("공용 키 파일")

    if not any(os.getenv(name) for name in API_KEY_NAMES):
        raise ConfigError(
            "Gemini API 키가 없습니다. 프로젝트 .env에 GEMINI_API_KEY를 넣거나, "
            "SHARED_ENV_FILE로 키가 있는 파일을 지정하세요 (.env.example 참고)"
        )
    return loaded


def _self_check() -> None:
    """키가 준비됐는지 확인한다. 키 값은 출력하지 않고 있는지 없는지만 보여준다."""
    loaded = load_env()
    print("읽은 파일:", ", ".join(loaded))
    for name in API_KEY_NAMES:
        print(f"{name} 설정됨: {bool(os.getenv(name))}")
    print("LANGSMITH_PROJECT:", os.getenv("LANGSMITH_PROJECT"))


if __name__ == "__main__":   # 직접 실행했을 때만 확인 코드를 돌린다
    _self_check()
