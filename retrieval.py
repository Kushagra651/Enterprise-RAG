from typing import List, Dict, Any, Optional
from embeddings import EmbeddingGenerator
from qdrant_manager import QdrantManager
from document_processor import DocumentProcessor
import ollama


class RAGRetrieval:
    def __init__(self):
        self.embedder = EmbeddingGenerator()
        self.qdrant = QdrantManager()
        self.llm_model = "mistral:7b"
        self._verify_llm()
    
    def _verify_llm(self):
        """Check if Mistral model is available"""
        try:
            # Simple test to verify model exists
            ollama.generate(model=self.llm_model, prompt="test", options={"num_predict": 1})
            print(f"✅ LLM model '{self.llm_model}' is ready")
        except Exception as e:
            print(f"❌ Error: Model '{self.llm_model}' not found")
            print(f"Run: ollama pull {self.llm_model}")
            raise e
    
    def ingest_documents(self, data_dir: str = "data"):
        """Process and ingest all documents into Qdrant"""
        print("\n🔄 Starting document ingestion...")
        
        # Load and chunk documents
        processor = DocumentProcessor(chunk_size=800, chunk_overlap=200)
        chunks = processor.process_all_documents(data_dir)
        
        if not chunks:
            print("⚠️ No documents found to ingest")
            return
        
        # Generate embeddings
        print("\n🔄 Generating embeddings...")
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedder.generate_embeddings_batch(texts)
        
        # Prepare documents for Qdrant
        documents = []
        for chunk, embedding in zip(chunks, embeddings):
            documents.append({
                "embedding": embedding,
                "text": chunk["text"],
                "department": chunk["department"],
                "doc_type": chunk["doc_type"],
                "created_date": chunk["created_date"],
                "source_file": chunk["source_file"],
                "chunk_id": chunk["chunk_id"]
            })
        
        # Insert into Qdrant
        print("\n🔄 Inserting into Qdrant...")
        self.qdrant.insert_documents(documents)
        
        # Show stats
        stats = self.qdrant.get_stats()
        print(f"\n✅ Ingestion complete!")
        print(f"📊 Total chunks indexed: {stats['total_points']}")
    
    def query(
        self,
        question: str,
        department: Optional[str] = None,
        doc_type: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """Query the knowledge base with optional filters"""
        
        # Generate query embedding
        query_embedding = self.embedder.generate_embedding(question)
        
        # Build filters
        filters = {}
        if department:
            filters["department"] = department
        if doc_type:
            filters["doc_type"] = doc_type
        if date_from:
            filters["date_from"] = date_from
        if date_to:
            filters["date_to"] = date_to
        
        # Search Qdrant
        results = self.qdrant.search(
            query_vector=query_embedding,
            filters=filters if filters else None,
            limit=top_k
        )
        
        if not results:
            return {
                "answer": "I couldn't find any relevant information in the knowledge base.",
                "sources": [],
                "filters_applied": filters
            }
        
        # Generate answer with LLM
        answer = self._generate_answer(question, results)
        
        return {
            "answer": answer,
            "sources": results,
            "filters_applied": filters
        }
    
    def _generate_answer(self, question: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Generate answer using Mistral with retrieved context"""
        
        # Build context from retrieved chunks
        context = "\n\n".join([
            f"[Source: {chunk['source_file']}]\n{chunk['text']}"
            for chunk in context_chunks[:3]  # Use top 3 chunks
        ])
        
        # Create prompt
        prompt = f"""You are a helpful corporate knowledge assistant. Answer the question based on the provided context from company documents.

Context:
{context}

Question: {question}

Instructions:
- Answer based ONLY on the provided context
- Be concise and professional
- If the context doesn't contain enough information, say so
- Cite the source documents when relevant

Answer:"""
        
        # Generate response
        try:
            response = ollama.generate(
                model=self.llm_model,
                prompt=prompt,
                options={
                    "temperature": 0.3,
                    "num_predict": 500
                }
            )
            return response['response'].strip()
        except Exception as e:
            return f"Error generating response: {e}"
    
    def compare_filtered_vs_unfiltered(self, question: str, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Compare retrieval with and without filters for evaluation"""
        
        # Query with filters
        filtered_results = self.query(question, **filters)
        
        # Query without filters
        unfiltered_results = self.query(question)
        
        return {
            "question": question,
            "filtered": {
                "answer": filtered_results["answer"],
                "num_sources": len(filtered_results["sources"]),
                "sources": filtered_results["sources"]
            },
            "unfiltered": {
                "answer": unfiltered_results["answer"],
                "num_sources": len(unfiltered_results["sources"]),
                "sources": unfiltered_results["sources"]
            },
            "filters_applied": filters
        }


# Test the retrieval system
if __name__ == "__main__":
    rag = RAGRetrieval()
    
    # Ingest documents
    rag.ingest_documents("data")
    
    print("\n" + "="*60)
    print("TESTING QUERIES")
    print("="*60)
    
    # Test query without filters
    print("\n1️⃣ Query without filters:")
    result = rag.query("What is the remote work policy?")
    print(f"\nAnswer: {result['answer']}")
    print(f"\nSources: {len(result['sources'])} documents")
    
    # Test query with department filter
    print("\n" + "="*60)
    print("2️⃣ Query with department filter (HR only):")
    result = rag.query(
        "What is the remote work policy?",
        department="Hr"
    )
    print(f"\nAnswer: {result['answer']}")
    print(f"\nSources: {len(result['sources'])} documents")
    for source in result['sources'][:2]:
        print(f"  - {source['source_file']} (Score: {source['score']:.3f})")
    
    # Test query with multiple filters
    print("\n" + "="*60)
    print("3️⃣ Query with multiple filters (Engineering + SOP):")
    result = rag.query(
        "What are the deployment procedures?",
        department="Engineering",
        doc_type="SOP"
    )
    print(f"\nAnswer: {result['answer']}")
    print(f"\nSources: {len(result['sources'])} documents")
    for source in result['sources'][:2]:
        print(f"  - {source['source_file']} (Score: {source['score']:.3f})")