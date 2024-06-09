import mesop.labs as mel
import anthropic

client = anthropic.Anthropic()
model = "claude-3-haiku-20240307"


def get_response(input: str, history: list[mel.ChatMessage]):

    messages = [{"role": "system", "content": "必ず日本語で回答してください。"}]
    for message in history:
        messages.append(
            {
                "role": "user" if message.role == "user" else "assistant",
                "content": message.content,
            }
        )
    messages.append({"role": "user", "content": input})

    with client.messages.stream(
        model=model,
        max_tokens=1000,  # 出力上限
        temperature=0.0,  # 0.0-1.0
        system="",  # 必要ならシステムプロンプトを設定
        messages=messages,
    ) as stream:
        for chunk in stream.text_stream:
            yield chunk
