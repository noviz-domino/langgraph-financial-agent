"""터미널에서 사용자 입력을 계속 받아 그래프를 실행한다.

루프는 두 단계로 나눈다 — 그래프 돌리기(run_turn) / 결과 보여주기(show_result).
Step 5에서 interrupt(승인 대기)가 생기면 show_result에 "멈췄으면 승인 받기"만 끼워 넣는다.
(지금은 interrupt가 없어서 그 길을 확인할 수 없으므로 미리 만들지 않는다)

결정 사항 (devlog/2026-09-28 참고)
    종료는 정해진 단어로만 (EXIT_WORDS) — LLM을 거치지 않는다. 종료는 은행 업무가 아니라 프로그램 조작이다
    Ctrl+C·Ctrl+D도 조용히 끝낸다. 빈 입력은 LLM을 부르지 않고 다시 받는다

단독 실행:  uv run python src/main.py   (질문마다 Gemini 1회 호출)
"""

import logging
import uuid

from langchain_core.messages import HumanMessage

from graph import build_graph

EXIT_WORDS = {"종료", "exit", "quit"}

# TODO(Step 5): 재시작 시 진행 중이던 업무가 있으면 확인하고, decision은 비운 뒤 다시 승인받기
#               (명세: "이전 승인만으로 변경을 자동 실행하지 않는다")


def run_turn(graph, config: dict, text: str) -> dict:
    """사용자 말 하나로 그래프를 한 바퀴 돌리고 최종 State를 돌려준다."""
    return graph.invoke({"messages": [HumanMessage(content=text)]}, config)


def show_result(state: dict) -> None:
    """그래프 결과를 화면에 보여준다. Step 5: 여기서 interrupt 여부를 보고 승인을 받는다."""
    print(state["messages"][-1].content)


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")
    graph = build_graph()
    config = {"configurable": {"thread_id": f"cli-{uuid.uuid4().hex[:8]}"}}   # 실행할 때마다 새 대화

    print("은행 업무 도우미입니다. 끝내려면 '종료'를 입력하세요.")
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
        show_result(run_turn(graph, config, text))
    print("이용해 주셔서 감사합니다.")


if __name__ == "__main__":
    main()
