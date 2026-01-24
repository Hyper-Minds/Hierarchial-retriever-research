# Hybrid Legal Document Retriever

## Overview  
This project implements a **Hybrid Legal Document Retrieval System** designed to improve the accuracy and relevance of judgment retrieval from large-scale legal corpora. The system combines **summary-level semantic relevance** with **fine-grained chunk-level matching** to preserve both high-level context and detailed legal nuances.

By leveraging this hybrid strategy, the retriever significantly enhances search quality for legal professionals, enabling faster and more reliable access to pertinent case law.

---

## Key Features

- **Hybrid Retrieval Architecture**  
  Integrates:
  - Document-level summaries for global contextual relevance  
  - Chunk-level embeddings for precise legal phrase and clause matching  

- **Context-Preserving Ranking**  
  Maintains semantic coherence across long judgments and orders.

- **Performance Improvement**  
  Achieved a **42% increase in Recall@10**, ensuring more relevant judgments appear in the top search results.

- **Scalable Design**  
  Optimized for large legal databases with thousands of lengthy, structured documents.

---

## System Architecture

1. **Preprocessing**
   - Judgment segmentation into semantic chunks
   - Automatic summary generation for each case
   - Embedding generation for both summaries and chunks

2. **Retrieval Pipeline**
   - Query embedding generation
   - Parallel similarity search on:
     - Summary embeddings (coarse-grained relevance)
     - Chunk embeddings (fine-grained relevance)
   - Hybrid score fusion and reranking

3. **Ranking & Output**
   - Weighted combination of summary and chunk relevance
   - Top-k judgments returned with most contextually aligned passages

---

## Evaluation

| Metric     | Improvement |
|-------------|-------------|
| Recall@10   | +42%        |
| Precision   | Improved contextual accuracy |
| Relevance   | Better coverage of legal nuances |

---

## Tech Stack

- **Language Models:** Sentence Transformers / LLMs  
- **Vector Store:** FAISS / Chroma / Pinecone (configurable)  
- **Backend:** Python, FastAPI  
- **Embedding Models:** BGE / Instructor / OpenAI (pluggable)

---

## Use Cases

- Legal research for advocates and judges  
- Case law similarity search  
- Precedent retrieval for argument drafting  
- Intelligent court information systems

---

## Future Enhancements

- Cross-encoder reranking for final result refinement  
- Citation-aware retrieval  
- Temporal relevance weighting  
- Multilingual Indian legal corpus support

---

## Contributors

Developed by **Tejeswar** as part of an AI-powered legal research initiative.
