import ollama
from typing import List, Union


class EmbeddingGenerator:
    def __init__(self, model: str = "nomic-embed-text"):
        self.model = model
        self._verify_model()
    
    def _verify_model(self):
        """Check if model is available in Ollama"""
        try:
            # Test embedding generation
            ollama.embeddings(model=self.model, prompt="test")
            print(f"✅ Embedding model '{self.model}' is ready")
        except Exception as e:
            print(f"❌ Error: Model '{self.model}' not found")
            print(f"Run: ollama pull {self.model}")
            raise e
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        try:
            response = ollama.embeddings(
                model=self.model,
                prompt=text
            )
            return response['embedding']
        except Exception as e:
            print(f"Error generating embedding: {e}")
            raise
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        embeddings = []
        total = len(texts)
        
        for idx, text in enumerate(texts, 1):
            if idx % 10 == 0:
                print(f"Processing {idx}/{total} embeddings...")
            
            embedding = self.generate_embedding(text)
            embeddings.append(embedding)
        
        print(f"✅ Generated {len(embeddings)} embeddings")
        return embeddings


# Test the embedding generator
if __name__ == "__main__":
    embedder = EmbeddingGenerator()
    
    # Test single embedding
    test_text = "This is a test document about Python programming."
    embedding = embedder.generate_embedding(test_text)
    print(f"✅ Embedding dimension: {len(embedding)}")
    print(f"✅ First 5 values: {embedding[:5]}")
    
    # Test batch embeddings
    test_texts = [
        "Engineering document about API design",
        "HR policy about remote work",
        "Guide for code reviews"
    ]
    embeddings = embedder.generate_embeddings_batch(test_texts)
    print(f"✅ Generated {len(embeddings)} embeddings")