"""데이터 무결성 검사.

JSON 데이터가 "은행 장부로서 말이 되는지"를 규칙별로 검사한다.
초기 데이터를 만들 때뿐 아니라, 나중에 이체 기능을 만든 뒤
"이체 후에도 장부가 맞는가?"를 확인하는 데에도 재사용한다.

단독 실행:  uv run python src/integrity.py                     (data/initial_data.json 검사)
            uv run python src/integrity.py data/bank_data.json (다른 파일 검사)
다른 모듈:  from integrity import check_integrity
"""

import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

# 한국 표준시(UTC+9).
# Windows에는 시간대 데이터베이스가 없어서 ZoneInfo("Asia/Seoul")가 실패할 수 있다.
# 한국은 서머타임이 없으므로 "항상 +9시간"인 고정 오프셋으로 충분하다.
KST = timezone(timedelta(hours=9))

# 이 파일(src/integrity.py)의 위치를 기준으로 경로를 잡는다
# → 어느 폴더에서 실행하든 항상 같은 파일을 찾는다
DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "initial_data.json"

# 검사 규칙: 키(코드에서 쓰는 이름) → 설명(사람이 읽는 문장)
RULES = {
    "balance_matches": "1. 계좌 잔액 = 입금 합계 - 출금 합계 (모든 계좌는 0원에서 시작)",
    "never_negative":  "2. 시간순으로 따라가도 잔액이 음수가 된 적 없음",
    "owner_matches":   "3. 거래의 소유자 = 그 거래가 속한 계좌의 소유자",
    "card_on_account": "4. 카드 결제는 그 계좌에 연결된 카드로만",
    "ids_valid":       "5. ID 중복 없음, 가리키는 대상이 실제로 존재",
    "values_valid":    "6. 금액은 0보다 큰 정수, 거래 유형은 deposit/withdrawal",
    "no_future":       "7. 기준 시각보다 미래의 거래 없음",
}

# ID 중복을 검사할 대상: (최상위 키, 그 안에서 ID 역할을 하는 필드 이름)
ID_FIELDS = [
    ("accounts", "account_id"),
    ("cards", "card_id"),
    ("transactions", "transaction_id"),
    ("addresses", "address_id"),
    ("bills", "bill_id"),
]


def _is_int(value) -> bool:
    """정수인지 확인. bool은 int의 하위 타입이라(True == 1) 따로 걸러낸다."""
    return isinstance(value, int) and not isinstance(value, bool)


def check_integrity(data: dict, now: datetime | None = None) -> dict[str, list[str]]:
    """규칙별 위반 내용 목록을 돌려준다. 모든 목록이 비어 있으면 통과.

    now: '미래 거래'를 판단하는 기준 시각. 안 넘기면 실행 시점의 현재 시각(KST).
         테스트에서 특정 날짜를 흉내 내고 싶을 때 이 값을 넘긴다.
    """
    now = now or datetime.now(KST)
    errors = {key: [] for key in RULES}   # 규칙마다 빈 목록을 하나씩 준비

    # ── 규칙 5: 컬렉션별 ID 중복 ─────────────────────────────
    for collection, id_field in ID_FIELDS:
        ids = [item[id_field] for item in data[collection]]
        for dup in sorted({i for i in ids if ids.count(i) > 1}):
            errors["ids_valid"].append(f"{collection}에 {dup} 중복")

    # ID로 바로 찾아 쓰기 위한 사전 (예: accounts["acc_001"] → 생활비 계좌 정보)
    accounts = {a["account_id"]: a for a in data["accounts"]}
    cards = {c["card_id"]: c for c in data["cards"]}

    # ── 규칙 5: 카드가 가리키는 계좌가 실제로 있는가 ──────────
    for card in data["cards"]:
        if card["account_id"] not in accounts:
            errors["ids_valid"].append(f"{card['card_id']}가 없는 계좌 {card['account_id']}를 가리킴")

    # ── 규칙 6: 잔액(0 이상)과 청구 금액(0 초과) ──────────────
    for acc in data["accounts"]:
        if not (_is_int(acc["balance"]) and acc["balance"] >= 0):
            errors["values_valid"].append(f"{acc['account_id']} 잔액이 {acc['balance']!r}")
    for bill in data["bills"]:
        if not (_is_int(bill["amount"]) and bill["amount"] > 0):
            errors["values_valid"].append(f"{bill['bill_id']} 금액이 {bill['amount']!r}")

    # ── 거래를 시간순으로 따라가며 규칙 1·2·3·4·6·7 검사 ─────
    running = defaultdict(int)   # 계좌별 누적 잔액. 처음엔 전부 0원 (0원 시작 규칙)
    ordered = sorted(data["transactions"], key=lambda t: datetime.fromisoformat(t["occurred_at"]))

    for tx in ordered:
        tx_id, acc_id = tx["transaction_id"], tx["account_id"]

        account = accounts.get(acc_id)
        if account is None:                                      # 규칙 5
            errors["ids_valid"].append(f"{tx_id}가 없는 계좌 {acc_id}를 가리킴")
            continue                                             # 계좌가 없으면 나머지 검사가 불가능

        if tx["owner_id"] != account["owner_id"]:                # 규칙 3
            errors["owner_matches"].append(
                f"{tx_id}: 거래 소유자 {tx['owner_id']}, 계좌 소유자 {account['owner_id']}"
            )

        if tx["card_id"] is not None:                            # 규칙 4·5 (카드 결제일 때만)
            card = cards.get(tx["card_id"])
            if card is None:
                errors["ids_valid"].append(f"{tx_id}가 없는 카드 {tx['card_id']}를 가리킴")
            elif card["account_id"] != acc_id:
                errors["card_on_account"].append(f"{tx_id}: {tx['card_id']}는 {acc_id}에 연결된 카드가 아님")

        if datetime.fromisoformat(tx["occurred_at"]) > now:      # 규칙 7
            errors["no_future"].append(f"{tx_id}: {tx['occurred_at']}")

        amount, tx_type = tx["amount"], tx["type"]
        if not (_is_int(amount) and amount > 0) or tx_type not in ("deposit", "withdrawal"):   # 규칙 6
            errors["values_valid"].append(f"{tx_id}: 금액 {amount!r}, 유형 {tx_type!r}")
            continue                                             # 잘못된 값으로 잔액 계산을 오염시키지 않음

        running[acc_id] += amount if tx_type == "deposit" else -amount
        if running[acc_id] < 0:                                  # 규칙 2
            errors["never_negative"].append(f"{tx_id} 직후 {acc_id} 잔액 {running[acc_id]:,}원")

    # ── 규칙 1: 따라가서 계산한 잔액이 저장된 잔액과 같은가 ───
    for acc_id, account in accounts.items():
        if running[acc_id] != account["balance"]:
            errors["balance_matches"].append(
                f"{acc_id}({account['nickname']}): 거래로 계산한 잔액 {running[acc_id]:,}원, "
                f"저장된 잔액 {account['balance']:,}원"
            )

    return errors


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH   # 인자로 파일을 주면 그 파일을 검사
    data = json.loads(path.read_text(encoding="utf-8"))
    errors = check_integrity(data)

    for key, description in RULES.items():
        status = "OK  " if not errors[key] else "FAIL"
        print(f"[{status}] {description}")
        for message in errors[key]:
            print(f"         - {message}")

    failed = sum(1 for messages in errors.values() if messages)
    print(f"결과: {len(RULES) - failed}/{len(RULES)} 통과 ({path.name})")
    sys.exit(1 if failed else 0)   # 실패하면 종료 코드 1 → 다른 도구(CI 등)가 실패를 알아챌 수 있음


if __name__ == "__main__":   # 직접 실행했을 때만 main()을 부른다 (다른 파일에서 import 할 때는 안 부름)
    main()
