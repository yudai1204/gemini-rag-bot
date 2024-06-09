from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.core import PromptTemplate
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.llms.gemini import Gemini

# from llama_index.llms.openai import OpenAI

# LLMの準備
# OpenAIを使用しても良い
Settings.llm = Gemini(
    model_name="models/gemini-1.5-flash",
)

# インデックスの読み込み
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)


custom_prompt = PromptTemplate(
    """\
Use Japanese for response.
Given a conversation (between Human and Assistant) and a follow up message from Human, \
rewrite the message to be a standalone question that captures all relevant context \
from the conversation.

<Chat History>
{chat_history}

<Follow Up Message>
{question}

<Standalone question>
"""
)

# list of `ChatMessage` objects
custom_chat_history = [
    ChatMessage(
        role=MessageRole.USER,
        content="地下の部屋には何がある？日本語で答えてください。",
    ),
    ChatMessage(
        role=MessageRole.ASSISTANT,
        content="地下の部屋には、巨大な円柱型の水槽とテーブルがあるようです。水槽は緑色の液体で満たされ、中に人型の何かが入っているとのことです。テーブルの上には実験レポートのようなものがあるようです。",
    ),
]

query_engine = index.as_query_engine()
chat_engine = CondenseQuestionChatEngine.from_defaults(
    query_engine=query_engine,
    condense_question_prompt=custom_prompt,
    chat_history=custom_chat_history,
    verbose=True,
)

# 質問応答
# print(chat_engine.chat("実験レポートには何が書かれていますか？"))
streaming_response = chat_engine.stream_chat(
    "実験レポートには何が書かれていますか？詳細に教えてください。"
)

for token in streaming_response.response_gen:
    print(token, end="")
