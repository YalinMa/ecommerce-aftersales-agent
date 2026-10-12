"""售后政策知识库：切分、索引、检索。

RAG（检索增强生成）的三步：
  1. 切分：把 knowledge/*.md 按"## 条款"切成小段（chunk），每段一个编号，如 R-01。
  2. 索引：离线算好每段的向量（scripts/build_index.py），存到 data/kb_index.json。
  3. 检索：用户提问时，找出最相关的几段交给模型，模型据此回答并引用编号。

支持三种检索方式，便于对比（scripts/eval_retrieval.py）：
  - bm25       关键词匹配。本地计算、零成本，擅长精确词（"专票""运费险"），不懂同义词
  - embedding  向量语义匹配。需调用 embedding 接口，擅长同义改写（"刻字的笔能退吗"→定制商品）
  - hybrid     两者结果用 RRF 融合，通常最稳
"""

import hashlib
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from mall import ROOT

KB_DIR = ROOT / "knowledge"
INDEX_PATH = ROOT / "data" / "kb_index.json"

# 可用的向量模型：厂商 → (模型, key 的环境变量, 接入点)。商城自己管理，不依赖客服 Agent 的配置
EMBEDDERS = {
    "qwen": "text-embedding-v4",
    "openai": "text-embedding-3-small",
}
_EMBEDDER_ENDPOINTS = {
    "qwen": ("DASHSCOPE_API_KEY", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    "openai": ("OPENAI_API_KEY", None),
}


def has_embedding_key(provider: str) -> bool:
    return bool(os.getenv(_EMBEDDER_ENDPOINTS[provider][0]))


@dataclass
class Chunk:
    id: str        # 条款编号，如 R-01
    title: str     # 条款标题
    doc: str       # 所属文档，如"退货与换货"
    content: str

    def text(self) -> str:
        """参与检索的文本：标题 + 正文。标题里的关键词信息量很大，要一起索引。"""
        return f"{self.doc} {self.title}\n{self.content}"


# ---------- 切分 ----------

_HEADING = re.compile(r"^##\s+([A-Z]-\d+)\s+(.+)$")


def load_chunks(kb_dir: Path = KB_DIR) -> list[Chunk]:
    chunks = []
    for path in sorted(kb_dir.glob("*.md")):
        doc, current, body = path.stem, None, []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("# "):
                doc = line[2:].strip()
            elif m := _HEADING.match(line):
                if current:
                    chunks.append(Chunk(*current, doc, "\n".join(body).strip()))
                current, body = (m.group(1), m.group(2).strip()), []
            elif current:
                body.append(line)
        if current:
            chunks.append(Chunk(*current, doc, "\n".join(body).strip()))
    return chunks


def fingerprint(chunks: list[Chunk]) -> str:
    """知识库内容的指纹。文档改了而索引没重建时，用它发现并提示。"""
    return hashlib.sha256("".join(c.text() for c in chunks).encode()).hexdigest()[:16]


# ---------- BM25 ----------

def tokenize(text: str) -> list[str]:
    """中文按单字 + 相邻双字切分，英文和数字按整词。

    不依赖分词库：双字组合能覆盖大部分中文词（"退货""运费""专票"），单字兜底。
    """
    tokens = []
    for run in re.findall(r"[一-鿿]+|[a-zA-Z0-9]+", text.lower()):
        if run[0].isascii():
            tokens.append(run)
        else:
            tokens.extend(run)
            tokens.extend(run[i:i + 2] for i in range(len(run) - 1))
    return tokens


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(d)) for d in docs]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avg_len = sum(self.lens) / len(self.lens)
        df = Counter(t for tf in self.tfs for t in tf)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        out = []
        for tf, length in zip(self.tfs, self.lens):
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * length / self.avg_len))
            out.append(s)
        return out


# ---------- 向量 ----------

def embedding_client(provider: str):
    """调用 OpenAI 兼容的 embeddings 接口。"""
    import openai

    if provider not in EMBEDDERS:
        raise SystemExit(f"不支持的向量模型厂商 {provider}，可选：{', '.join(EMBEDDERS)}")
    key_env, default_url = _EMBEDDER_ENDPOINTS[provider]
    if not os.getenv(key_env):
        raise SystemExit(f"未配置 {key_env}，无法使用 {provider} 向量模型")
    base_url = os.getenv(f"{provider.upper()}_BASE_URL") or default_url
    return openai.OpenAI(api_key=os.getenv(key_env), base_url=base_url)


def embed(provider: str, texts: list[str], batch: int = 10) -> list[list[float]]:
    client, model, vectors = embedding_client(provider), EMBEDDERS[provider], []
    for i in range(0, len(texts), batch):   # 部分厂商单次最多 10 条
        resp = client.embeddings.create(model=model, input=texts[i:i + batch])
        vectors.extend(_normalize(d.embedding) for d in resp.data)
    return vectors


def _normalize(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))   # 向量已归一化，点积即余弦相似度


# ---------- 检索器 ----------

@dataclass
class Hit:
    chunk: Chunk
    score: float


class Retriever:
    def __init__(self, method: str | None = None, embedder: str | None = None):
        self.chunks = load_chunks()
        self.bm25 = BM25([c.text() for c in self.chunks])
        self.embedder = embedder or os.getenv("KB_EMBEDDER", "qwen")
        self.vectors = None
        self._query_cache: dict[str, list[float]] = {}

        index = json.loads(INDEX_PATH.read_text(encoding="utf-8")) if INDEX_PATH.exists() else None
        if index and index["fingerprint"] != fingerprint(self.chunks):
            raise SystemExit("知识库文档已修改，请重新运行：python scripts/build_index.py")
        if index and self.embedder in index["embeddings"]:
            self.vectors = index["embeddings"][self.embedder]["vectors"]

        # 默认混合检索；没有向量索引时退回关键词检索
        self.method = method or os.getenv("KB_METHOD") or ("hybrid" if self.vectors else "bm25")
        if self.method in ("embedding", "hybrid") and not self.vectors:
            raise SystemExit(f"没有 {self.embedder} 的向量索引，请先运行：python scripts/build_index.py")

    def _embed_query(self, query: str) -> list[float]:
        if query not in self._query_cache:
            self._query_cache[query] = embed(self.embedder, [query])[0]
        return self._query_cache[query]

    def _rank(self, scores: list[float]) -> list[int]:
        return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    def search(self, query: str, top_k: int = 4) -> list[Hit]:
        if self.method == "bm25":
            scores = self.bm25.scores(query)
        elif self.method == "embedding":
            q = self._embed_query(query)
            scores = [_dot(q, v) for v in self.vectors]
        else:
            # RRF（倒数排名融合）：只看两路结果各自的名次，不需要把两种分数换算到同一尺度
            q = self._embed_query(query)
            ranks_bm25 = self._rank(self.bm25.scores(query))
            ranks_vec = self._rank([_dot(q, v) for v in self.vectors])
            scores = [0.0] * len(self.chunks)
            for ranking in (ranks_bm25, ranks_vec):
                for pos, i in enumerate(ranking):
                    scores[i] += 1 / (60 + pos + 1)
        order = self._rank(scores)[:top_k]
        return [Hit(self.chunks[i], round(scores[i], 4)) for i in order]


_default: Retriever | None = None


def default_retriever() -> Retriever:
    global _default
    if _default is None:
        _default = Retriever()
    return _default
