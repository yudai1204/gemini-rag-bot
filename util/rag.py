from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.core import PromptTemplate
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.llms.gemini import Gemini
import mesop.labs as mel

# LLMの準備
Settings.llm = Gemini(
    model_name="models/gemini-1.5-flash",
)

# インデックスの読み込み
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)

custom_prompt = PromptTemplate(
    """\
Please use Japanese and provide detailed answers.
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


query_engine = index.as_query_engine()


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

    streaming_response = chat_engine.stream_chat(input)

    for token in streaming_response.response_gen:
        yield token
