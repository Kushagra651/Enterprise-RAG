from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, Range
from typing import List, Dict, Any, Optional
import uuid


class QdrantManager:
    def __init__(self, collection_name: str = "enterprise_docs"):
        # Persistent mode
        self.client = QdrantClient(path="./qdrant_data")
        self.collection_name = collection_name
        self._setup_collection()
    
    def _setup_collection(self):
        """Create collection with vector config"""
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=768,  # nomic-embed-text dimension
                    distance=Distance.COSINE
                )
            )
            print(f"✅ Collection '{self.collection_name}' created")
        except Exception as e:
            print(f"⚠️ Collection already exists or error: {e}")
    
    def insert_documents(self, documents: List[Dict[str, Any]]):
        """Insert document chunks with embeddings and metadata"""
        points = []
        for doc in documents:
            point_id = str(uuid.uuid4())
            points.append(
                PointStruct(
                    id=point_id,
                    vector=doc["embedding"],
                    payload={
                        "text": doc["text"],
                        "department": doc["department"],
                        "doc_type": doc["doc_type"],
                        "created_date": doc["created_date"],
                        "source_file": doc["source_file"],
                        "chunk_id": doc.get("chunk_id", 0)
                    }
                )
            )
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print(f"✅ Inserted {len(documents)} chunks into Qdrant")
    
    def search(
        self,
        query_vector: List[float],
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Search with optional metadata filters"""
        qdrant_filter = self._build_filter(filters) if filters else None
        
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=qdrant_filter,
            limit=limit,
            with_payload=True
        )
        
        return [
            {
                "score": result.score,
                "text": result.payload["text"],
                "department": result.payload["department"],
                "doc_type": result.payload["doc_type"],
                "source_file": result.payload["source_file"]
            }
            for result in results
        ]
    
    def _build_filter(self, filters: Dict[str, Any]) -> Filter:
        """Build Qdrant filter from dict"""
        conditions = []
        
        # Department filter (supports single or multiple)
        if filters.get("department"):
            dept = filters["department"]
            if isinstance(dept, list):
                # Multiple departments
                conditions.append(
                    FieldCondition(
                        key="department",
                        match=MatchValue(any=dept)
                    )
                )
            else:
                # Single department
                conditions.append(
                    FieldCondition(
                        key="department",
                        match=MatchValue(value=dept)
                    )
                )
        
        # Doc type filter (supports single or multiple)
        if filters.get("doc_type"):
            dtype = filters["doc_type"]
            if isinstance(dtype, list):
                # Multiple doc types
                conditions.append(
                    FieldCondition(
                        key="doc_type",
                        match=MatchValue(any=dtype)
                    )
                )
            else:
                # Single doc type
                conditions.append(
                    FieldCondition(
                        key="doc_type",
                        match=MatchValue(value=dtype)
                    )
                )
        
        # Date range filter
        if filters.get("date_from") or filters.get("date_to"):
            date_range = {}
            if filters.get("date_from"):
                date_range["gte"] = filters["date_from"]
            if filters.get("date_to"):
                date_range["lte"] = filters["date_to"]
            
            conditions.append(
                FieldCondition(
                    key="created_date",
                    range=Range(**date_range)
                )
            )
        
        return Filter(must=conditions) if conditions else None
    
    def get_stats(self):
        """Get collection statistics"""
        info = self.client.get_collection(self.collection_name)
        return {
            "total_points": info.points_count,
            "vector_size": info.config.params.vectors.size
        }
    
    def delete_collection(self):
        """Delete the collection"""
        try:
            self.client.delete_collection(self.collection_name)
            print(f"✅ Collection '{self.collection_name}' deleted")
        except Exception as e:
            print(f"❌ Error deleting collection: {e}")


# Test the setup
if __name__ == "__main__":
    # Test initialization
    qm = QdrantManager()
    
    # Test insert
    test_docs = [
        {
            "embedding": [0.1] * 768,
            "text": "This is a test document about Python best practices.",
            "department": "Engineering",
            "doc_type": "Guide",
            "created_date": "2024-01-15",
            "source_file": "python_guide.pdf",
            "chunk_id": 0
        }
    ]
    qm.insert_documents(test_docs)
    
    # Test search without filters
    print("\n1️⃣ Search without filters:")
    results = qm.search(
        query_vector=[0.1] * 768,
        limit=3
    )
    print(f"✅ Found {len(results)} results")
    
    # Test search with single department
    print("\n2️⃣ Search with single department:")
    results = qm.search(
        query_vector=[0.1] * 768,
        filters={"department": "Engineering"},
        limit=3
    )
    print(f"✅ Found {len(results)} results")
    
    # Test search with multiple departments
    print("\n3️⃣ Search with multiple departments:")
    results = qm.search(
        query_vector=[0.1] * 768,
        filters={"department": ["Engineering", "Hr"]},
        limit=3
    )
    print(f"✅ Found {len(results)} results")
    
    # Test stats
    stats = qm.get_stats()
    print(f"\n✅ Stats: {stats}")