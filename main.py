import mesop as me
import mesop.labs as mel

# from util.gemini import get_response

from util.llama_index_rag import get_response


@me.page(
    security_policy=me.SecurityPolicy(
        allowed_iframe_parents=["http://localhost:32123"]
    ),
    path="/",
    title="Mesop Demo Chat",
)
def page():
    mel.chat(get_response, title="Mesop Demo Chat", bot_user="Mesop Bot")
