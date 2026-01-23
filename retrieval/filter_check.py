from collections import defaultdict
from typing import List, Dict
from langchain_qdrant import QdrantVectorStore
from vectorstore.set_up_collections import get_summary_store, get_coarse_chunk_store
from qdrant_client import models

coarse_chunk_store = get_coarse_chunk_store()

query = "Cases where appeal of High Court upheld by Supreme Court"

coarse_chunk_results = coarse_chunk_store.similarity_search_with_score(
                query,
                k=5,
                filter = models.Filter(
                    should=[
                        models.FieldCondition(
                            key="metadata.doc_id",
                            match=models.MatchValue(value="019ba122-e277-7113-aced-4fcac60a537e")
                        )
                    ]
                )
            )

for doc, score in coarse_chunk_results:
    print(f"Score: {score} \nDocument: {doc.metadata}\n")