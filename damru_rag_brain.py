#!/usr/bin/env python3
"""
damru_rag_brain.py
=================
RAG (Retrieval-Augmented Generation) Brain for Damru AI

Integrates daily learning corpus into semantic search + knowledge retrieval.
- Ingests JSONL corpus from learn/daily/
- Builds embedding index (in-memory or vector DB)
- Fast semantic retrieval for chat queries
- Quality scoring and relevance ranking

Built by Shiva AI for Damru
"""

import json
import os
import sys
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# Try to use efficient libraries, fall back to basic
try:
    from sentence_transformers import SentenceTransformer, util
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

def log(msg):
    print(f"[rag-brain] {msg}", flush=True)

class DamruRAGBrain:
    """
    RAG Brain: Retrieves relevant knowledge from daily corpus
    """
    
    def __init__(self, 
                 corpus_dir: str = "./learn/daily",
                 model_name: str = "all-MiniLM-L6-v2",
                 top_k: int = 5):
        """
        Initialize RAG Brain
        
        Args:
            corpus_dir: Path to daily JSONL files
            model_name: SentenceTransformer model for embeddings
            top_k: Number of top results to retrieve
        """
        self.corpus_dir = corpus_dir
        self.model_name = model_name
        self.top_k = top_k
        
        # Data storage
        self.records: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index: Optional[Any] = None  # FAISS index if available
        
        # Model
        self.encoder = None
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                self.encoder = SentenceTransformer(model_name)
                log(f"Loaded encoder: {model_name}")
            except Exception as e:
                log(f"Failed to load encoder: {e}")
        
        # Load corpus
        self._load_corpus()
        if self.records:
            self._build_index()
    
    def _load_corpus(self):
        """Load all JSONL files from corpus directory"""
        if not os.path.exists(self.corpus_dir):
            log(f"Corpus dir not found: {self.corpus_dir}")
            return
        
        files = sorted([f for f in os.listdir(self.corpus_dir) if f.endswith('.jsonl')])
        log(f"Found {len(files)} corpus files")
        
        for fname in files[-7:]:  # Load last 7 days
            fpath = os.path.join(self.corpus_dir, fname)
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                rec = json.loads(line)
                                self.records.append(rec)
                            except json.JSONDecodeError:
                                pass
                log(f"Loaded {fname}: {len(self.records)} total records")
            except Exception as e:
                log(f"Error loading {fname}: {e}")
        
        log(f"Total corpus size: {len(self.records)} records")
    
    def _build_index(self):
        """Build embedding index"""
        if not self.records:
            log("No records to index")
            return
        
        if not self.encoder:
            log("No encoder available, skipping indexing")
            return
        
        try:
            # Prepare texts (question + answer for better context)
            texts = []
            for rec in self.records:
                q = rec.get('question', '')
                a = rec.get('answer', '')
                text = f"{q} {a}"[:512]  # Truncate to avoid too long texts
                texts.append(text)
            
            log(f"Encoding {len(texts)} records...")
            embeddings = self.encoder.encode(texts, show_progress_bar=True, convert_to_numpy=True)
            self.embeddings = embeddings
            
            # Try to build FAISS index for fast retrieval
            if HAS_FAISS and embeddings.shape[0] > 0:
                dimension = embeddings.shape[1]
                self.index = faiss.IndexFlatIP(dimension)
                self.index.add(embeddings.astype('float32'))
                log(f"Built FAISS index: {self.index.ntotal} vectors")
            
            log("Index ready!")
        
        except Exception as e:
            log(f"Failed to build index: {e}")
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most relevant records for query
        
        Args:
            query: User query
            top_k: Override default top_k
        
        Returns:
            List of relevant records with scores
        """
        if not self.records:
            return []
        
        k = top_k or self.top_k
        k = min(k, len(self.records))
        
        try:
            if self.encoder and self.embeddings is not None:
                query_embedding = self.encoder.encode(query, convert_to_numpy=True)
                
                if self.index:
                    # Use FAISS for fast retrieval
                    distances, indices = self.index.search(
                        np.array([query_embedding], dtype='float32'), k
                    )
                    results = []
                    for idx, score in zip(indices[0], distances[0]):
                        if idx >= 0:
                            rec = self.records[idx].copy()
                            rec['score'] = float(score)
                            results.append(rec)
                else:
                    # Fallback: cosine similarity
                    from sentence_transformers import util
                    scores = util.cos_sim(query_embedding, self.embeddings)[0]
                    top_results = np.argsort(scores.cpu().numpy())[::-1][:k]
                    results = []
                    for idx in top_results:
                        rec = self.records[idx].copy()
                        rec['score'] = float(scores[idx])
                        results.append(rec)
                
                return results
        
        except Exception as e:
            log(f"Retrieval error: {e}")
        
        # Fallback: keyword search
        return self._keyword_search(query, k)
    
    def _keyword_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Fallback keyword-based search"""
        query_words = set(query.lower().split())
        scores = []
        
        for i, rec in enumerate(self.records):
            q = (rec.get('question', '') or '').lower()
            a = (rec.get('answer', '') or '').lower()
            text = f"{q} {a}"
            text_words = set(text.split())
            
            overlap = len(query_words & text_words)
            if overlap > 0:
                scores.append((i, overlap))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in scores[:k]:
            rec = self.records[idx].copy()
            rec['score'] = float(score)
            results.append(rec)
        
        return results
    
    def augment_prompt(self, query: str, context_size: int = 3) -> str:
        """
        Create augmented prompt with retrieved context
        
        Args:
            query: Original user query
            context_size: Number of context items
        
        Returns:
            Augmented prompt with context
        """
        retrieved = self.retrieve(query, top_k=context_size)
        
        if not retrieved:
            return query
        
        context_lines = ["Context from Damru Knowledge Base:"]
        for i, rec in enumerate(retrieved, 1):
            q = rec.get('question', '')
            a = rec.get('answer', '')
            score = rec.get('score', 0)
            context_lines.append(f"\n[Context {i}] (relevance: {score:.2f})")
            context_lines.append(f"Q: {q}")
            context_lines.append(f"A: {a[:200]}...")  # Truncate long answers
        
        context_lines.append(f"\n---\n\nUser Query: {query}")
        return "\n".join(context_lines)
    
    def stats(self) -> Dict[str, Any]:
        """Get brain stats"""
        return {
            "corpus_size": len(self.records),
            "has_embeddings": self.embeddings is not None,
            "has_faiss_index": self.index is not None,
            "embedding_model": self.model_name if self.encoder else None,
        }


def main():
    """Demo / test"""
    brain = DamruRAGBrain()
    
    log("Brain initialized")
    log(f"Stats: {brain.stats()}")
    
    if brain.records:
        # Test query
        test_query = "How to be productive?"
        log(f"\nTest query: {test_query}")
        results = brain.retrieve(test_query, top_k=3)
        for i, rec in enumerate(results, 1):
            log(f"\n[Result {i}]")
            log(f"Q: {rec.get('question')}")
            log(f"A: {rec.get('answer', '')[:100]}...")
            log(f"Score: {rec.get('score'):.3f}")
        
        # Test augmented prompt
        aug_prompt = brain.augment_prompt(test_query)
        log(f"\nAugmented prompt:\n{aug_prompt}")


if __name__ == "__main__":
    main()
