"""터미널에서 사용자 입력을 계속 받아 그래프를 실행한다.

루프는 두 단계로 나눈다 — 그래프 돌리기(run_turn) / 결과 보여주기(show_result).
그래프가 승인 화면에서 멈추면(interrupt) show_result가 알려주고, 다음 입력은 새 요청이 아니라
멈춘 그래프를 이어 가는 답(Command(resume=...))으로 보낸다.

결정 사항 (devlog/2026-09-28 참고)
    종료는 정해진 단어로만 (EXIT_WORDS) — LLM을 거치지 않는다. 종료는 은행 업무가 아니라 프로그램 조작이다
    Ctrl+C·Ctrl+D도 조용히 끝낸다. 빈 입력은 LLM을 부르지 않고 다시 받는다

단독 실행:  uv run python src/main.py   (질문마다 Gemini 1회 호출)
"""

import logging
import uuid

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from config import load_env
from graph import build_graph

EXIT_WORDS = {"종료", "exit", "quit"}

# TODO(3차, SqliteSaver로 바꿀 때): 재시작 시 진행 중이던 업무가 있으면 확인하고, decision은 비운 뒤 다시 승인받기
#               (명세: "이전 승인만으로 변경을 자동 실행하지 않는다"). 지금은 InMemorySaver라 끄면 대기 중 요청이 사라진다


def run_turn(graph, config: dict, text: str, waiting: bool) -> dict:
    """사용자 말 하나로 그래프를 돌린다.

    waiting=False  새 요청 → START부터 한 바퀴
    waiting=True   승인 화면에서 멈춰 있음 → 이 말은 승인 답. 멈춘 자리부터 이어 간다
    """
    if waiting:
        return graph.invoke(Command(resume=text), config)
    return graph.invoke({"messages": [HumanMessage(content=text)]}, config)


def show_result(state: dict) -> bool:
    """그래프 결과를 화면에 보여준다. 승인을 기다리며 멈춰 있으면 True."""
    interrupts = state.get("__interrupt__")                     # 멈췄으면 invoke 결과에 이 칸이 생긴다
    if interrupts:
        print(interrupts[0].value["prompt"])
        return True
    print(state["messages"][-1].content)
    return False


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")
    logging.getLogger("google_genai").setLevel(logging.ERROR)   # AFC 안내 경고 — 동작과 무관해서 사용자 화면에서 숨긴다
    load_env()                                                  # 첫 invoke 전에 — LangSmith는 Tracing 여부를 처음 한 번만 읽고 기억한다
    graph = build_graph()
    config = {"configurable": {"thread_id": f"cli-{uuid.uuid4().hex[:8]}"}}   # 실행할 때마다 새 대화

    print("은행 업무 도우미입니다. 예) '내 계좌 전부 보여줘', '생활비에서 저축으로 10만 원 보내줘', '생활비 카드 잠가줘'")
    print("끝내려면 '종료'를 입력하세요.")
    waiting = False                                             # 승인 답을 기다리는 중인가
    while True:
        try:
            text = input("\n> ").strip()
        except (KeyboardInterrupt, EOFError):                   # Ctrl+C / Ctrl+D (Windows는 Ctrl+Z 엔터)
            print()
            break
        if not text:
            continue
        if text.lower() in EXIT_WORDS:
            break
        waiting = show_result(run_turn(graph, config, text, waiting))
    print("이용해 주셔서 감사합니다.")


if __name__ == "__main__":
    main()
