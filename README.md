# Gemini と mesop を使った RAG チャットボットの実装

## 事前準備

```bash
pip install -r requirements.txt
```

```bash
cp .envrc.temp .envrc
```

OpenAI および Gemini の API キーを取得し、.envrc に記述してください。

## 実行前の準備

1. `data`の中に学習に利用したいデータを入れてください。
1. `genIndex.py`を実行して、事前に embedding を計算してください。
1. `storage`ディレクトリができていることを確認し、次章に移ってください。

## 実行

```bash
mesop main.py
```

[http://localhost:32123](http://localhost:32123) にアクセスしてください。
