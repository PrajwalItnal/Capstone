"""
RAG Engine: Embedding-based Question Retrieval using FAISS + ChromaDB.
Implements semantic skill matching, question tagging, and retrieval-augmented generation
as required by the TCS Capstone project specification.
"""

import os
import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path

# ─── Graceful imports ──────────────────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    chromadb = None

# ─── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
CHROMA_DIR = BASE_DIR / "chroma_store"
CHROMA_DIR.mkdir(exist_ok=True)

# ─── Embedding model ───────────────────────────────────────────────────────────
_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
_embed_model = None

def get_embed_model():
    global _embed_model
    if _embed_model is None and SENTENCE_TRANSFORMERS_AVAILABLE:
        try:
            _embed_model = SentenceTransformer(_EMBED_MODEL_NAME)
        except Exception as e:
            print(f"[RAG] Could not load embedding model: {e}")
    return _embed_model


def embed_texts(texts: List[str]) -> Optional[np.ndarray]:
    """Return float32 numpy array of shape (N, D) or None if unavailable."""
    model = get_embed_model()
    if model is None or not texts:
        return None
    try:
        vecs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return vecs.astype(np.float32)
    except Exception as e:
        print(f"[RAG] Embedding error: {e}")
        return None


# ════════════════════════════════════════════════════════════════════════════════
# FAISS Index
# ════════════════════════════════════════════════════════════════════════════════

class FAISSIndex:
    """Flat cosine-similarity FAISS index for nearest-neighbour retrieval."""

    def __init__(self):
        self.index = None
        self.documents: List[Dict] = []
        self.dimension: int = 384

    def build(self, documents: List[Dict], text_field: str = "text"):
        if not FAISS_AVAILABLE:
            return False
        texts = [d.get(text_field, "") for d in documents]
        vecs = embed_texts(texts)
        if vecs is None:
            return False
        self.dimension = vecs.shape[1]
        faiss.normalize_L2(vecs)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(vecs)
        self.documents = documents
        return True

    def search(self, query: str, top_k: int = 5) -> List[Tuple[float, Dict]]:
        if self.index is None or self.index.ntotal == 0:
            return []
        qvec = embed_texts([query])
        if qvec is None:
            return []
        faiss.normalize_L2(qvec)
        scores, idxs = self.index.search(qvec, min(top_k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx >= 0:
                results.append((float(score), self.documents[idx]))
        return results


# ════════════════════════════════════════════════════════════════════════════════
# ChromaDB Persistent Store
# ════════════════════════════════════════════════════════════════════════════════

_chroma_client = None
_chroma_collection = None

def get_chroma_collection(collection_name: str = "question_bank"):
    global _chroma_client, _chroma_collection
    if not CHROMA_AVAILABLE:
        return None
    try:
        if _chroma_client is None:
            _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _chroma_collection = _chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        return _chroma_collection
    except Exception as e:
        print(f"[RAG] ChromaDB error: {e}")
        return None


# ════════════════════════════════════════════════════════════════════════════════
# RAG Engine
# ════════════════════════════════════════════════════════════════════════════════

class RAGEngine:
    """
    Embedding-based retrieval engine for competency taxonomy and question bank.
    Implements semantic search (FAISS), persistent storage (ChromaDB), and
    multi-label tag recommendations as required by the capstone specification.
    """

    def __init__(self):
        self.question_index = FAISSIndex()
        self.taxonomy_index = FAISSIndex()
        self.taxonomy_docs: List[Dict] = []
        self.question_bank: List[Dict] = []
        self._indexed = False

    # ── Indexing ──────────────────────────────────────────────────────────────

    def index_question_bank(self, questions: List[Dict]):
        """Index question dicts {question/text, competency, difficulty, ...} into FAISS + ChromaDB."""
        self.question_bank = questions
        docs = []
        for q in questions:
            q_text = q.get("question") or q.get("text", "")
            combined = f"{q_text} {q.get('competency', '')} {q.get('subskill', '')} {q.get('tags', '')}"
            doc = {**q, "text": combined.strip(), "question": q_text}
            docs.append(doc)
        self.question_index.build(docs, text_field="text")

        col = get_chroma_collection("question_bank")
        if col is not None:
            try:
                ids = [str(q.get("id", i)) for i, q in enumerate(questions)]
                texts = [d["text"] for d in docs]
                metadatas = []
                for q in questions:
                    meta = {}
                    for k, v in q.items():
                        if k != "text":
                            meta[k] = str(v) if not isinstance(v, (str, int, float, bool)) else v
                    metadatas.append(meta)
                col.upsert(ids=ids, documents=texts, metadatas=metadatas)
            except Exception as e:
                print(f"[RAG] ChromaDB upsert error: {e}")

        self._indexed = True

    def index_taxonomy(self, taxonomy: Dict):
        """Index the competency taxonomy for semantic skill mapping."""
        docs = []
        for skill, info in taxonomy.items():
            aliases = " ".join(info.get("aliases", []))
            text = f"{skill} {info.get('canonical', '')} {aliases}"
            docs.append({
                "skill": skill,
                "canonical": info.get("canonical", skill),
                "category": info.get("category", ""),
                "text": text.strip(),
            })
        self.taxonomy_docs = docs
        self.taxonomy_index.build(docs, text_field="text")

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve_questions(
        self,
        query: str,
        competency: Optional[str] = None,
        difficulty: Optional[str] = None,
        top_k: int = 5,
        min_score: float = 0.3,
    ) -> List[Dict]:
        """Retrieve semantically similar questions, optionally filtered."""
        if self.question_index.index is not None:
            results = self.question_index.search(query, top_k=top_k * 3)
            filtered = []
            for score, doc in results:
                if score < min_score:
                    continue
                if competency and doc.get("competency", "").lower() != competency.lower():
                    continue
                if difficulty and doc.get("difficulty", "").lower() != difficulty.lower():
                    continue
                filtered.append({**doc, "_score": round(score, 4)})
                if len(filtered) >= top_k:
                    break
            if filtered:
                return filtered

        col = get_chroma_collection("question_bank")
        if col is not None and col.count() > 0:
            try:
                where = {}
                if competency:
                    where["competency"] = competency
                kwargs = {"query_texts": [query], "n_results": min(top_k, col.count())}
                if where:
                    kwargs["where"] = where
                res = col.query(**kwargs)
                docs = res.get("documents", [[]])[0]
                metas = res.get("metadatas", [[]])[0]
                dists = res.get("distances", [[]])[0]
                return [
                    {**m, "text": d, "_score": round(1 - dist, 4)}
                    for d, m, dist in zip(docs, metas, dists)
                ]
            except Exception as e:
                print(f"[RAG] ChromaDB query error: {e}")

        return self._keyword_fallback(query, competency, difficulty, top_k)

    def _keyword_fallback(self, query, competency, difficulty, top_k):
        keywords = set(query.lower().split())
        results = []
        for q in self.question_bank:
            text = (q.get("question") or q.get("text", "")).lower()
            score = sum(1 for kw in keywords if kw in text) / max(len(keywords), 1)
            if competency and q.get("competency", "").lower() != competency.lower():
                score *= 0.5
            results.append({**q, "_score": round(score, 4)})
        results.sort(key=lambda x: x["_score"], reverse=True)
        return results[:top_k]

    # ── Semantic Tagging ──────────────────────────────────────────────────────

    def recommend_tags(self, question_text: str, top_k: int = 3) -> List[Dict]:
        """Recommend top_k competency tags for a question using cosine similarity."""
        if self.taxonomy_index.index is None:
            return []
        results = self.taxonomy_index.search(question_text, top_k=top_k)
        return [
            {
                "competency": doc.get("canonical", doc.get("skill", "")),
                "category": doc.get("category", ""),
                "confidence": round(score, 3),
            }
            for score, doc in results
            if score > 0.25
        ]

    def detect_wrong_tags(
        self, question_text: str, assigned_competency: str, threshold: float = 0.35
    ) -> Dict:
        """Detect if a question is assigned to the wrong competency."""
        tags = self.recommend_tags(question_text, top_k=1)
        if not tags:
            return {"is_wrong": False, "suggested_competency": assigned_competency, "confidence": 0.0}
        top_tag = tags[0]
        is_wrong = (
            top_tag["competency"].lower() != assigned_competency.lower()
            and top_tag["confidence"] > threshold
        )
        return {
            "is_wrong": is_wrong,
            "suggested_competency": top_tag["competency"],
            "confidence": top_tag["confidence"],
        }

    # ── Competency Mapping ────────────────────────────────────────────────────

    def map_skill_to_competency(self, skill_text: str) -> Optional[Dict]:
        """Map a raw extracted skill to canonical competency via semantic similarity."""
        if self.taxonomy_index.index is None:
            return None
        results = self.taxonomy_index.search(skill_text, top_k=1)
        if results:
            score, doc = results[0]
            if score > 0.25:
                return {
                    "canonical": doc.get("canonical", ""),
                    "category": doc.get("category", ""),
                    "confidence": round(score, 3),
                }
        return None

    # ── Duplicate Detection ───────────────────────────────────────────────────

    def find_duplicates(
        self, questions: List[Dict], similarity_threshold: float = 0.85
    ) -> List[Tuple[int, int, float]]:
        """Find pairs of questions that are too similar (potential duplicates)."""
        texts = [q.get("question") or q.get("text", "") for q in questions]
        vecs = embed_texts(texts)
        if vecs is None or len(vecs) < 2:
            return []
        if FAISS_AVAILABLE and faiss:
            faiss.normalize_L2(vecs)
        else:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vecs = vecs / norms
        duplicates = []
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                sim = float(np.dot(vecs[i], vecs[j]))
                if sim >= similarity_threshold:
                    duplicates.append((i, j, round(sim, 3)))
        return duplicates

    # ── Status ────────────────────────────────────────────────────────────────

    def status(self) -> Dict:
        q_count = self.question_index.index.ntotal if self.question_index.index else 0
        t_count = self.taxonomy_index.index.ntotal if self.taxonomy_index.index else 0
        col = get_chroma_collection()
        chroma_count = 0
        if col:
            try:
                chroma_count = col.count()
            except Exception:
                pass
        return {
            "sentence_transformers": SENTENCE_TRANSFORMERS_AVAILABLE,
            "faiss": FAISS_AVAILABLE,
            "chromadb": CHROMA_AVAILABLE,
            "questions_indexed_faiss": q_count,
            "taxonomy_indexed_faiss": t_count,
            "questions_in_chromadb": chroma_count,
            "model": _EMBED_MODEL_NAME,
        }

    get_status = status


# ─── Singleton ─────────────────────────────────────────────────────────────────
_rag_engine: Optional[RAGEngine] = None

def get_rag_engine() -> RAGEngine:
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine
