## gas_agent_voice.py
## 목적: "주유소" 세계관 대화형 에이전트 + 음성 입출력(MVP)
## 약어 병기: ASR(Automatic Speech Recognition, 자동 음성 인식),
##            TTS(Text-to-Speech, 음성 합성),
##            LLM(Large Language Model, 대형 언어 모델)

import json, requests, sys, os
from typing import Dict, Any
from pydantic import BaseModel, Field

## ==== 설정 ====
OLLAMA_URL = "http://localhost:11434/api/generate"
VOSK_MODEL_DIR = "./models/vosk-model-small-ko-0.22"  ## ASR 모델 경로
SAMPLE_RATE = 16000
RECORD_SEC = 5  ## 말하기 시간(초): 엔터 누르면 4초 녹음

## ==== LLM 프롬프트 (주유소 전용) ====
SYSTEM_PROMPT = """
너는 '주유소' 세계관 전용 대화형 비서다. 사용자의 말을 이해해 의도(intent)와 인자(arguments)를 추출하고,
부족하면 짧게 되물어본다(clarify). 그리고 실행할 도구(tool_name)와 사용자에게 들려줄 한두 문장(assistant_say)을 함께 만든다.
'주유소'와 무관한 요청은 스몰톡(chitchat)으로 답한다.

반드시 아래 JSON '하나만' 출력한다:
{
  "intent": "<thirst|hunger|select_fuel|authorize_payment|start_fueling|stop_fueling|check_price|car_wash|receipt|find_restroom|buy_item|chitchat|none>",
  "confidence": <0~1 number>,
  "tool_name": "<select_fuel|authorize_payment|start_fueling|stop_fueling|check_price|car_wash|receipt|find_restroom|buy_item|clarify|null>",
  "arguments": { ... },
  "assistant_say": "<한국어 한두 문장>"
}

도구 규칙(주요 예):
- select_fuel: {"fuel_type": "gasoline|diesel|premium|lpg|ev"}
- authorize_payment: {"method":"card|cash|mobile", "prepay_amount":"<숫자+원>", "pump_id":"선택"}
- start_fueling: {"pump_id":"숫자/문자", "fuel_type":"...", "target":"<10L|20000원 등, 선택>"}
- stop_fueling: {"pump_id":"..."}
- check_price: {"fuel_type":"gasoline|diesel|premium|lpg"}
- car_wash: {"wash_type":"basic|premium"}
- receipt: {"method":"paper|sms|email", "contact":"선택"}
- find_restroom: {}
- buy_item: {"item":"물|과자|아이스커피 등", "qty":"선택"}
- 정보가 부족하면 tool_name='clarify'와 {"question":"..."} 사용.
- 스몰톡은 intent='chitchat', tool_name=null 로 두고 assistant_say만 생성.
- assistant_say는 항상 한국어로 짧고 자연스럽게. JSON 외 텍스트 금지.
"""

FEWSHOTS = """
User: "경유 3만원"
{"intent":"start_fueling","confidence":0.86,"tool_name":"start_fueling",
 "arguments":{"pump_id":"","fuel_type":"diesel","target":"30000원"},
 "assistant_say":"경유로 3만원 주유 도와드릴게요. 주유기 번호 알려주세요."}

User: "2번 주유기 카드 결제할게"
{"intent":"authorize_payment","confidence":0.9,"tool_name":"authorize_payment",
 "arguments":{"method":"card","prepay_amount":"","pump_id":"2"},
 "assistant_say":"2번 주유기 카드 결제 진행할게요."}

User: "세차 돼?"
{"intent":"car_wash","confidence":0.8,"tool_name":"clarify",
 "arguments":{"question":"기본과 프리미엄 중 어떤 세차를 원하시나요?"},
 "assistant_say":"가능해요. 기본과 프리미엄 중 어느 걸로 할까요?"}
"""

## ==== 도구 필수 인자 ====
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

## ==== LLM 응답 모델 ====
class ConvoFrame(BaseModel):
    intent: str
    confidence: float = Field(ge=0, le=1)
    tool_name: str | None
    arguments: dict
    assistant_say: str

## ==== LLM 호출 ====
def call_llm(user_text: str, model: str = "llama3.1") -> ConvoFrame:
    prompt = f"{SYSTEM_PROMPT}\n{FEWSHOTS}\nUser: {json.dumps(user_text, ensure_ascii=False)}\n"
    payload = {"model": model, "prompt": prompt, "format": "json", "options": {"temperature": 0.2}}
    r = requests.post(OLLAMA_URL, json=payload, timeout=120, stream=True)
    r.raise_for_status()
    text = ""
    for line in r.iter_lines():
        if not line: continue
        data = json.loads(line.decode("utf-8"))
        if "response" in data: text += data["response"]
        if data.get("done"): break
    obj = json.loads(text)
    return ConvoFrame(**obj)

## ==== 필수 인자 체크 ====
def missing_required(tool_name: str, args: dict) -> list[str]:
    req = TOOLS.get(tool_name, {}).get("required", [])
    return [k for k in req if k not in args or args[k] in (None, "")]

## ==== 도구 실행(여기서 ROS2/Isaac으로 바꿔 끼우면 실기 연동) ====
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

## ==== TTS: pyttsx3 (오프라인) ====
def speak(text: str):
    ## pyttsx3는 OS 음성 엔진을 사용한다. Linux는 espeak-ng 권장.
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 185)  ## 말하기 속도
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"[TTS WARN] 음성 합성 실패: {e}")

## ==== ASR: Vosk (오프라인) ====
def transcribe_seconds(sec: int = RECORD_SEC, sr: int = SAMPLE_RATE) -> str:
    ## 엔터 후 sec초 동안 녹음 → 한국어 인식
    ## PortAudio 사운드 장치 연결 필요: sounddevice 사용
    try:
        from vosk import Model, KaldiRecognizer
        import sounddevice as sd
        import numpy as np
    except Exception as e:
        print(f"[ASR WARN] 모듈 누락: {e}")
        return ""

    if not os.path.isdir(VOSK_MODEL_DIR):
        print(f"[ASR WARN] Vosk 모델 경로가 없습니다: {VOSK_MODEL_DIR}")
        return ""

    try:
        model = Model(VOSK_MODEL_DIR)
        rec = KaldiRecognizer(model, sr)
        rec.SetWords(True)

        print(f"[REC] {sec}초 녹음 시작... (말하세요)")
        audio = sd.rec(int(sec * sr), samplerate=sr, channels=1, dtype="int16")
        sd.wait()
        print("[REC] 녹음 종료")

        ## 바이트로 변환 후 인식
        buf = audio.tobytes()
        if rec.AcceptWaveform(buf):
            res = json.loads(rec.Result())
        else:
            res = json.loads(rec.FinalResult())

        text = res.get("text", "").strip()
        return text
    except Exception as e:
        print(f"[ASR WARN] 인식 실패: {e}")
        return ""

## ==== 메인(음성 + 텍스트 하이브리드) ====
def main():
    print("주유소 음성 어시스턴트 시작 (종료: q)")
    print("엔터 = 음성 입력(4초), 't' 입력 = 텍스트 입력, 'q' = 종료")

    while True:
        key = input("> ").strip().lower()
        if key == "q":
            break

        if key == "t":
            user = input("YOU(text) > ").strip()
        else:
            user = transcribe_seconds(sec=RECORD_SEC)  ## 엔터면 음성 인식
            if not user:
                print("…(인식 실패/무음) 다시 시도하세요.")
                continue
            print(f"YOU(voice) > {user}")

        ## LLM 호출
        try:
            frame = call_llm(user)
        except Exception as e:
            print(f"[ERR] LLM 호출 실패: {e}")
            print(" - 'ollama run llama3.1'을 한 번 실행해 모델이 준비됐는지 확인")
            return

        ## 봇 멘트 + 말하기
        print(f"BOT > {frame.assistant_say}")
        speak(frame.assistant_say)

        ## 스몰톡이면 행동 없음
        if not frame.tool_name:
            continue

        ## 필수 인자 빠지면 되물음(clarify)
        miss = missing_required(frame.tool_name, frame.arguments)
        if miss and frame.tool_name != "clarify":
            q = f"{', '.join(miss)} 값이 필요해요. 알려주세요."
            print(f"[ASK] {q}")
            speak(q)
            continue

        ## 행동 실행
        run_tool(frame.tool_name, **frame.arguments)

if __name__ == "__main__":
    main()

