from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.core import PromptTemplate
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.llms.gemini import Gemini
from llama_index.llms.openai import OpenAI

import mesop.labs as mel

# 最終的な回答の生成はGeminiの方が高精度
useGemini = True

# LLMの準備
if useGemini:
    Settings.llm = Gemini(
        model_name="models/gemini-1.5-flash",
    )
else:
    Settings.llm = OpenAI(
        model_name="gpt-4o",
    )

# インデックスの読み込み
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)

custom_prompt = PromptTemplate(
    """\
Please use Japanese. and provide detailed long answers.
Given a conversation (between Human and Assistant) and a follow up message from Human, \
rewrite the message to be a standalone question that captures all relevant context \
from the conversation.
必ず日本語で答えてください。

<Chat History>
{chat_history}

<Follow Up Message>
{question}

<Standalone question>
"""
)


query_engine = index.as_query_engine(similarity_top_k=10)  # 類似度の高い10件を取得


def get_response(input: str, history: list[mel.ChatMessage]):
    custom_chat_history = []
    for message in history:
        custom_chat_history.append(
            ChatMessage(
                role=(
                    MessageRole.USER
                    if message.role == "user"
                    else MessageRole.ASSISTANT
                ),
                content=message.content,
            )
        )

    chat_engine = CondenseQuestionChatEngine.from_defaults(
        query_engine=query_engine,
        condense_question_prompt=custom_prompt,
        chat_history=custom_chat_history,
        verbose=True,
    )

    streaming_response = chat_engine.stream_chat("日本語で回答してください。" + input)

    for token in streaming_response.response_gen:
        yield token
