import os
import google.generativeai as genai
import mesop.labs as mel

# API-KEYの設定
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)

chat = None


def start_chat():
    model_name = "gemini-1.5-flash"
    config = {
        "temperature": 0.6,  # 生成するテキストのランダム性を制御
        "top_k": 5,  # 生成に使用するトップkトークンを制御
        # "max_output_tokens": 512,  # 最大出力トークン数を指定`
    }

    model = genai.GenerativeModel(
        model_name,
        generation_config=config,
        system_instruction=[  # System
            "Use Japanese for response.",
        ],
    )
    return model.start_chat()


def get_response(input: str, history: list[mel.ChatMessage]):
    global chat
    if chat is None:
        chat = start_chat()

    response = chat.send_message(input, stream=True)

    for chunk in response:
        yield chunk.text
