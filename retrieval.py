from typing import List, Dict, Any, Optional
from embeddings import EmbeddingGenerator
from qdrant_manager import QdrantManager
from document_processor import DocumentProcessor
from query_decomposition import QueryDecomposer
import ollama


class RAGRetrieval:
    def __init__(self):
        self.embedder = EmbeddingGenerator()
        self.qdrant = QdrantManager()
        self.decomposer = QueryDecomposer()  # NEW: Added decomposer
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
    
    def query_with_decomposition(
        self,
        question: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        NEW: Query with automatic decomposition for complex questions
        """
        
        print(f"\n🔍 Analyzing query: {question}")
        
        # Step 1: Decompose query
        decomposition = self.decomposer.decompose_query(question)
        
        # Step 2: Execute sub-queries
        sub_results = []
        
        for idx, sub_q in enumerate(decomposition['sub_queries'], 1):
            print(f"\n📌 Sub-query {idx}/{len(decomposition['sub_queries'])}: {sub_q['sub_query']}")
            
            # Build filters from decomposition
            filters = {}
            if sub_q['department']:
                filters['department'] = sub_q['department']
            if sub_q['doc_type']:
                filters['doc_type'] = sub_q['doc_type']
            
            # Execute query
            result = self.query(
                question=sub_q['sub_query'],
                **filters,
                top_k=top_k
            )
            
            sub_results.append({
                'sub_query': sub_q['sub_query'],
                'department': sub_q['department'],
                'doc_type': sub_q['doc_type'],
                'answer': result['answer'],
                'sources': result['sources']
            })
        
        # Step 3: Synthesize if multiple sub-queries
        if decomposition['needs_decomposition'] and len(sub_results) > 1:
            print(f"\n🔄 Synthesizing {len(sub_results)} answers...")
            final_answer = self.decomposer.synthesize_answers(question, sub_results)
        else:
            final_answer = sub_results[0]['answer']
        
        # Collect all unique sources
        all_sources = []
        seen_files = set()
        for result in sub_results:
            for source in result['sources']:
                if source['source_file'] not in seen_files:
                    all_sources.append(source)
                    seen_files.add(source['source_file'])
        
        return {
            "answer": final_answer,
            "sources": all_sources,
            "decomposition": decomposition,
            "sub_results": sub_results
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
    print("TESTING STANDARD QUERIES")
    print("="*60)
    
    # Test query without filters
    print("\n1️⃣ Query without filters:")
    result = rag.query("What is the remote work policy?")
    print(f"\nAnswer: {result['answer'][:200]}...")
    print(f"Sources: {len(result['sources'])} documents")
    
    # Test query with department filter
    print("\n" + "="*60)
    print("2️⃣ Query with department filter (HR only):")
    result = rag.query(
        "What is the remote work policy?",
        department="Hr"
    )
    print(f"\nAnswer: {result['answer'][:200]}...")
    print(f"Sources: {len(result['sources'])} documents")
    
    print("\n" + "="*60)
    print("TESTING QUERY DECOMPOSITION")
    print("="*60)
    
    # Test complex query with decomposition
    result = rag.query_with_decomposition(
        "What is the deployment process for remote engineering employees?"
    )
    
    print(f"\n✅ Final Answer:\n{result['answer'][:300]}...")
    print(f"\n📚 Total Sources: {len(result['sources'])}")
    print(f"\n🔍 Decomposition:")
    print(f"  Needs decomposition: {result['decomposition']['needs_decomposition']}")
    print(f"  Sub-queries: {len(result['sub_results'])}")
    for idx, sub in enumerate(result['sub_results'], 1):
        print(f"\n  {idx}. {sub['sub_query']}")
        print(f"     Dept: {sub['department']}, Type: {sub['doc_type']}")