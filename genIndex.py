from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex,
    SimpleDirectoryReader,
    VectorStoreIndex,
)
from llama_index.embeddings.gemini import GeminiEmbedding
import logging
import sys
import logging
import sys

# Geminiを使って埋め込みを作成するかどうか
# 基本的にはOpenAIのほうが高精度なため、Falseにしておく
USE_GEMINI = False

# ログレベルの設定
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)

# ドキュメントの読み込み
documents = SimpleDirectoryReader("data").load_data()

if USE_GEMINI:
    # 埋め込みモデルの準備
    Settings.embed_model = GeminiEmbedding(
        model_name="models/text-embedding-004",
    )

# インデックスの作成
index = VectorStoreIndex.from_documents(documents)

# インデックスの保存
index.storage_context.persist()
