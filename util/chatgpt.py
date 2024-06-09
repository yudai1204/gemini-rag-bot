from openai import OpenAI
import mesop.labs as mel

client = OpenAI()
model_name = "gpt-4o"


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

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        n=1,  # 出力の数
        max_tokens=1000,  # 出力上限
        temperature=0.5,  # 0.0-1.0
        stream=True,
    )

    for chunk in response:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
