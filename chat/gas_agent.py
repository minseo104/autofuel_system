## gas_agent.py
## 목적: "주유소" 세계관 전용 대화형 에이전트 (LLM: Large Language Model)
## 요구 패키지: requests, pydantic  /  런타임: Ollama(로컬 LLM)

import json, requests, sys
from typing import Dict, Any
from pydantic import BaseModel, Field

OLLAMA_URL = "http://localhost:11434/api/generate"

## ---- 시스템 프롬프트: 오직 '주유소' 맥락만 다룬다 ----
SYSTEM_PROMPT = """
너는 '주유소' 세계관 전용 대화형 비서다. 사용자의 말을 이해해 의도(intent)와 인자(arguments)를 추출하고,
필요하면 짧게 되물어본다(clarify). 그리고 실행할 도구(tool_name)와 사용자에게 들려줄 한두 문장(assistant_say)을 함께 만든다.
'주유소'와 무관한 요청은 스몰톡(chitchat)으로 답한다.

반드시 아래 JSON '하나만' 출력한다:
{
  "intent": "<thirst|hunger|select_fuel|authorize_payment|start_fueling|stop_fueling|check_price|car_wash|receipt|find_restroom|buy_item|chitchat|none>",
  "confidence": <0~1 number>,
  "tool_name": "<select_fuel|authorize_payment|start_fueling|stop_fueling|check_price|car_wash|receipt|find_restroom|buy_item|clarify|null>",
  "arguments": { ... },     // tool_name에 맞는 인자
  "assistant_say": "<한국어 한두 문장>"
}

도구 규칙(주요 예):
- select_fuel: {"fuel_type": "gasoline|diesel|premium|lpg|ev"}
- authorize_payment: {"method":"card|cash|mobile", "prepay_amount": "<숫자+원|달러, 선택>", "pump_id":"선택"}
- start_fueling: {"pump_id":"숫자/문자", "fuel_type":"...", "target":"<10L|20000원 등, 선택>"}
- stop_fueling: {"pump_id":"..."}
- check_price: {"fuel_type":"gasoline|diesel|premium|lpg"}
- car_wash: {"wash_type":"basic|premium"}
- receipt: {"method":"paper|sms|email", "contact":"선택"}
- find_restroom: {}
- buy_item: {"item":"물|과자|아이스커피 등", "qty":"선택"}
- 정보가 부족하거나 선택이 필요하면 tool_name='clarify'와 {"question":"..."} 사용.
- 스몰톡은 intent='chitchat', tool_name=null 로 두고 assistant_say만 생성.
- assistant_say는 항상 한국어로 짧고 자연스럽게. JSON 외 텍스트 금지.
"""

## ---- 몇-shot 예시 ----
FEWSHOTS = """
User: "휘발유 만원만 넣어줘"
{"intent":"start_fueling","confidence":0.85,"tool_name":"start_fueling",
 "arguments":{"pump_id":"","fuel_type":"gasoline","target":"10000원"},
 "assistant_say":"휘발유로 1만원 주유 도와드릴게요. 사용하실 주유기 번호 알려주세요."}

User: "2번 주유기 카드 결제할게"
{"intent":"authorize_payment","confidence":0.9,"tool_name":"authorize_payment",
 "arguments":{"method":"card","prepay_amount":"","pump_id":"2"},
 "assistant_say":"2번 주유기 카드 결제 진행할게요."}

User: "경유 가격이 지금 얼마야?"
{"intent":"check_price","confidence":0.88,"tool_name":"check_price",
 "arguments":{"fuel_type":"diesel"},
 "assistant_say":"경유 현재 가격을 확인해볼게요."}

User: "세차도 되나요?"
{"intent":"car_wash","confidence":0.8,"tool_name":"clarify",
 "arguments":{"question":"기본 세차와 프리미엄 중 어떤 걸 원하시나요?"},
 "assistant_say":"가능해요. 기본과 프리미엄 중 어느 걸로 할까요?"}

User: "영수증 문자로 보내줘"
{"intent":"receipt","confidence":0.85,"tool_name":"receipt",
 "arguments":{"method":"sms","contact":""},
 "assistant_say":"문자 영수증으로 처리할게요. 전화번호를 알려주세요."}

User: "화장실 어디야"
{"intent":"find_restroom","confidence":0.9,"tool_name":"find_restroom",
 "arguments":{},
 "assistant_say":"매장 오른쪽 복도 끝 화장실 이용하시면 됩니다."}

User: "아 배고픈데 과자 있어?"
{"intent":"buy_item","confidence":0.82,"tool_name":"buy_item",
 "arguments":{"item":"snack","qty":""},
 "assistant_say":"편의 매대에서 과자 준비해드릴게요."}

User: "안녕?"
{"intent":"chitchat","confidence":0.7,"tool_name":null,
 "arguments":{},
 "assistant_say":"안녕하세요. 주유나 세차, 결제 도와드릴까요?"}
"""

## ---- 도구 필수 인자 정의 ----
TOOLS: Dict[str, Dict[str, Any]] = {
    "select_fuel": {"required": ["fuel_type"]},
    "authorize_payment": {"required": ["method"]},
    "start_fueling": {"required": ["pump_id", "fuel_type"]},
    "stop_fueling": {"required": ["pump_id"]},
    "check_price": {"required": ["fuel_type"]},
    "car_wash": {"required": ["wash_type"]},
    "receipt": {"required": ["method"]},
    "find_restroom": {"required": []},
    "buy_item": {"required": ["item"]},
    "clarify": {"required": ["question"]}
}

## ---- LLM 응답 모델 ----
class ConvoFrame(BaseModel):
    intent: str
    confidence: float = Field(ge=0, le=1)
    tool_name: str | None
    arguments: dict
    assistant_say: str

## ---- Ollama 호출 ----
def call_llm(user_text: str, model: str = "llama3.1") -> ConvoFrame:
    prompt = f"{SYSTEM_PROMPT}\n{FEWSHOTS}\nUser: {json.dumps(user_text, ensure_ascii=False)}\n"
    payload = {"model": model, "prompt": prompt, "format": "json", "options": {"temperature": 0.2}}
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=120, stream=True)
        r.raise_for_status()
    except Exception as e:
        print(f"[ERR] Ollama 연결 실패: {e}")
        print("      1) 'ollama run llama3.1' 한 번 실행했는지 확인")
        print("      2) Ollama 서비스(포트 11434)가 켜져 있는지 확인")
        sys.exit(1)

    text = ""
    for line in r.iter_lines():
        if not line: continue
        data = json.loads(line.decode("utf-8"))
        if "response" in data: text += data["response"]
        if data.get("done"): break

    obj = json.loads(text)
    return ConvoFrame(**obj)

## ---- 필수 인자 검사 ----
def missing_required(tool_name: str, args: dict) -> list[str]:
    req = TOOLS.get(tool_name, {}).get("required", [])
    return [k for k in req if k not in args or args[k] in (None, "")]

## ---- 도구 실행(현재는 프린트; 추후 ROS2/Isaac 연동 지점) ----
def run_tool(tool_name: str, **kw) -> dict:
    if tool_name == "select_fuel":
        print(f"[ACT] SELECT_FUEL fuel_type={kw.get('fuel_type')}")
    elif tool_name == "authorize_payment":
        print(f"[ACT] AUTHORIZE_PAYMENT method={kw.get('method')} prepay={kw.get('prepay_amount','')} pump={kw.get('pump_id','')}")
    elif tool_name == "start_fueling":
        print(f"[ACT] START_FUELING pump={kw.get('pump_id')} fuel={kw.get('fuel_type')} target={kw.get('target','')}")
    elif tool_name == "stop_fueling":
        print(f"[ACT] STOP_FUELING pump={kw.get('pump_id')}")
    elif tool_name == "check_price":
        print(f"[ACT] CHECK_PRICE fuel={kw.get('fuel_type')}")
    elif tool_name == "car_wash":
        print(f"[ACT] CAR_WASH type={kw.get('wash_type')}")
    elif tool_name == "receipt":
        print(f"[ACT] RECEIPT method={kw.get('method')} contact={kw.get('contact','')}")
    elif tool_name == "find_restroom":
        print(f"[ACT] FIND_RESTROOM")
    elif tool_name == "buy_item":
        print(f"[ACT] BUY_ITEM item={kw.get('item')} qty={kw.get('qty','')}")
    elif tool_name == "clarify":
        print(f"[ASK] {kw.get('question','필요 정보가 있어요.')}")
    else:
        print(f"[WARN] Unknown tool: {tool_name}")
        return {"ok": False, "error": "unknown_tool"}
    return {"ok": True}

## ---- 메인 루프 ----
def main():
    print("주유소 어시스턴트를 시작합니다. (종료: q)")
    while True:
        user = input("YOU > ").strip()
        if user.lower() == "q": break

        frame = call_llm(user)
        print(f"BOT > {frame.assistant_say}")

        if not frame.tool_name:  ## 스몰톡
            continue

        miss = missing_required(frame.tool_name, frame.arguments)
        if miss and frame.tool_name != "clarify":
            run_tool("clarify", question=f"{', '.join(miss)} 값이 필요해요. 알려주세요.")
            continue

        run_tool(frame.tool_name, **frame.arguments)

if __name__ == "__main__":
    main()

