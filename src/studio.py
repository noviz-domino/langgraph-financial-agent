"""LangGraph Studio 입구. langgraph dev가 이 파일의 graph를 가져가 실행한다 (langgraph.json).

main.py(터미널)·web.py(브라우저)와 같은 역할의 입구다. 에이전트 코드는 그대로 두고 입구만 하나 더했다.
다른 입구와 다른 점은 셋:
  - 실행하는 주인이 우리 프로그램이 아니라 LangGraph 서버라서, 폴더 이름 없는 import가 되도록 src/를 찾는 목록에 넣는다
  - 서버가 멈춘 상태를 자기 저장소에 저장하므로 checkpointer 없이 compile한다
  - .env를 그래프보다 먼저 읽는다 (LangSmith가 Tracing 여부를 처음 한 번만 확인하고 기억함)
장부는 main.py와 같은 작업용 복사본(bank_data.json)을 쓴다.

실행:  PYTHONUTF8=1 uv run langgraph dev   (API 호출 없음. Studio에서 말을 보낼 때마다 Gemini 1회)
  - PYTHONUTF8=1: 한국어 Windows에서 langgraph-api가 자기 파일을 cp949로 읽다가 실패한다 (인코딩을 지정하지 않음)
  - 서버가 멈춘 상태를 저장하는 폴더 .langgraph_api/는 git에 올리지 않는다
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import load_env  # noqa: E402 — 위에서 찾는 목록을 고친 뒤에 불러와야 한다

load_env()

from graph import build_graph  # noqa: E402

graph = build_graph(use_checkpointer=False)
