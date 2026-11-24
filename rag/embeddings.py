"""bge-small-en-v1.5 (quantized ONNX) as a LangChain Embeddings class.

Runs on CPU with onnxruntime, so the demo needs no embedding API key and the
same model embeds documents at build time and questions at query time. (A
different model at query time than at index time is one of the most common
reasons a RAG bot "suddenly" retrieves garbage.)
"""
from pathlib import Path

import numpy as np
import onnxruntime as ort
from langchain_core.embeddings import Embeddings
from tokenizers import Tokenizer

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "bge-small-en-v1.5"
MODEL_NAME = "BAAI/bge-small-en-v1.5 (int8 ONNX)"
# bge models expect this instruction on queries only, never on passages.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class BgeSmallEmbeddings(Embeddings):
    def __init__(self, model_dir: Path = MODEL_DIR, max_len: int = 512):
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.sess = ort.InferenceSession(str(model_dir / "model_quantized.onnx"), opts, providers=["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.sess.get_inputs()}
        self.tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tok.enable_truncation(max_len)
        self.tok.enable_padding(pad_id=0, pad_token="[PAD]")

    def embed(self, texts: list[str], batch: int = 32) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch):
            enc = self.tok.encode_batch(texts[i : i + batch])
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.input_names:
                feed["token_type_ids"] = np.zeros_like(ids)
            hidden = self.sess.run(None, feed)[0]
            cls = hidden[:, 0]  # bge uses the [CLS] vector, then L2-normalises
            out.append(cls / np.linalg.norm(cls, axis=1, keepdims=True))
        return np.vstack(out).astype(np.float32) if out else np.zeros((0, 384), np.float32)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed(texts).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed([QUERY_PREFIX + text])[0].tolist()

    def query_vector(self, text: str) -> np.ndarray:
        return self.embed([QUERY_PREFIX + text])[0]
