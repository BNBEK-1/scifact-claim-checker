from __future__ import annotations

import json
import re
import tarfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

DATA_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
NLI_MODEL = "cross-encoder/nli-deberta-v3-base"


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def verdict_for_scores(support: float, contradict: float, threshold: float) -> str:
    if support >= threshold and contradict >= threshold and abs(support - contradict) < 0.12:
        return "MIXED_EVIDENCE"
    if max(support, contradict) < threshold:
        return "NOT_ENOUGH_EVIDENCE"
    return "SUPPORTS" if support > contradict else "CONTRADICTS"


def load_dataset(data_dir: str | Path = "data") -> tuple[list[dict], list[dict]]:
    data_dir = Path(data_dir)
    corpus_path = data_dir / "corpus.jsonl"
    dev_path = data_dir / "claims_dev.jsonl"
    if not corpus_path.exists() or not dev_path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        archive_path = data_dir / "scifact.tar.gz"
        if not archive_path.exists():
            temp_path = archive_path.with_suffix(".download")
            with urlopen(DATA_URL, timeout=60) as response, temp_path.open("wb") as out:
                out.write(response.read())
            temp_path.replace(archive_path)
        with tarfile.open(archive_path, "r:gz") as archive:
            for member in archive.getmembers():
                if member.isfile() and Path(member.name).name in {"corpus.jsonl", "claims_dev.jsonl", "claims_train.jsonl"}:
                    member.name = Path(member.name).name
                    archive.extract(member, data_dir)
    if not corpus_path.exists() or not dev_path.exists():
        raise FileNotFoundError("SciFact archive did not contain corpus.jsonl and claims_dev.jsonl.")
    corpus = [json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    dev = [json.loads(line) for line in dev_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for doc in corpus:
        abstract = doc.get("abstract", [])
        if isinstance(abstract, str):
            try:
                abstract = json.loads(abstract)
            except json.JSONDecodeError:
                abstract = [abstract]
        doc["abstract"] = abstract
        doc["text"] = " ".join([doc.get("title", ""), *abstract]).strip()
    return corpus, dev

class ClaimChecker:
    def __init__(self, corpus: list[dict], dense_model: str = DENSE_MODEL, nli_model: str = NLI_MODEL):
        self.corpus = corpus
        self.bm25 = BM25Okapi([_tokens(doc["text"]) for doc in corpus])
        self.encoder = SentenceTransformer(dense_model)
        self.doc_vectors = self.encoder.encode(
            [doc["text"] for doc in corpus], normalize_embeddings=True,
            show_progress_bar=False, batch_size=64,
        )
        self.nli = CrossEncoder(nli_model, max_length=384)
        self.label_ids = self._get_label_ids()

    def _get_label_ids(self) -> dict[str, int]:
        labels = {str(v).lower(): int(k) for k, v in self.nli.model.config.id2label.items()}
        found = {}
        for name in ("entailment", "contradiction", "neutral"):
            for label, idx in labels.items():
                if name in label:
                    found[name] = idx
                    break
        if len(found) == 3:
            return found
        if self.nli.model.config.num_labels == 3:
            return {"contradiction": 0, "entailment": 1, "neutral": 2}
        raise RuntimeError(f"Could not identify NLI labels: {labels}")

    def retrieve(self, claim: str, method: str = "Hybrid", top_k: int = 10) -> list[dict]:
        query = _tokens(claim)
        bm_scores = np.asarray(self.bm25.get_scores(query), dtype=float)
        bm_order = np.argsort(-bm_scores)
        query_vec = self.encoder.encode([claim], normalize_embeddings=True)[0]
        dense_scores = self.doc_vectors @ query_vec
        dense_order = np.argsort(-dense_scores)
        if method == "BM25":
            order = bm_order[:top_k]
        elif method == "Dense":
            order = dense_order[:top_k]
        else:
            # Rank fusion keeps either retriever from dominating just because its scores use another scale.
            fused = np.zeros(len(self.corpus), dtype=float)
            for ranked in (bm_order, dense_order):
                for rank, idx in enumerate(ranked[:100], start=1):
                    fused[idx] += 1 / (60 + rank)
            order = np.argsort(-fused)[:top_k]
        return [{
            **self.corpus[int(idx)],
            "bm25_score": float(bm_scores[int(idx)]),
            "dense_score": float(dense_scores[int(idx)]),
        } for idx in order]

    def verify(self, claim: str, method: str = "Hybrid", top_k: int = 10, threshold: float = 0.55) -> dict:
        docs = self.retrieve(claim, method, top_k)
        pairs, refs = [], []
        for doc in docs:
            sentences = doc.get("abstract", [])
            if isinstance(sentences, str):
                sentences = [sentences]
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence:
                    pairs.append((sentence, claim))
                    refs.append((doc, sentence))
        if not pairs:
            return {"verdict": "NOT_ENOUGH_EVIDENCE", "support": 0.0, "contradict": 0.0, "evidence": [], "documents": docs}
        probs = np.asarray(self.nli.predict(pairs, apply_softmax=True, show_progress_bar=False))
        ent_i, con_i = self.label_ids["entailment"], self.label_ids["contradiction"]
        scored = []
        for (doc, sentence), row in zip(refs, probs):
            scored.append({
                "doc_id": str(doc["doc_id"]), "title": doc.get("title", "Untitled"),
                "sentence": sentence, "entailment": float(row[ent_i]),
                "contradiction": float(row[con_i]),
            })
        support = max((item["entailment"] for item in scored), default=0.0)
        contradict = max((item["contradiction"] for item in scored), default=0.0)
        verdict = verdict_for_scores(support, contradict, threshold)
        evidence = sorted(scored, key=lambda item: max(item["entailment"], item["contradiction"]), reverse=True)[:3]
        return {"verdict": verdict, "support": support, "contradict": contradict, "evidence": evidence, "documents": docs}
