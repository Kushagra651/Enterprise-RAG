import os
from typing import List, Dict, Any
from datetime import datetime
import re
from pathlib import Path


class DocumentProcessor:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def load_documents(self, data_dir: str = "data") -> List[Dict[str, Any]]:
        """Load all documents from engineering and hr folders"""
        documents = []
        
        for department in ["engineering", "hr"]:
            dept_path = os.path.join(data_dir, department)
            
            if not os.path.exists(dept_path):
                print(f"⚠️ Warning: {dept_path} not found")
                continue
            
            for filename in os.listdir(dept_path):
                if filename.endswith('.txt'):
                    filepath = os.path.join(dept_path, filename)
                    
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Extract metadata from content
                    metadata = self._extract_metadata(content, filename, department)
                    
                    documents.append({
                        "content": content,
                        "metadata": metadata,
                        "source_file": filename
                    })
        
        print(f"✅ Loaded {len(documents)} documents")
        return documents
    
    def _extract_metadata(self, content: str, filename: str, department: str) -> Dict[str, str]:
        """Extract metadata from document content and filename"""
        metadata = {
            "department": department.capitalize(),
            "doc_type": "Guide",  # Default
            "created_date": "2024-01-01",  # Default
            "source_file": filename
        }
        
        # Extract doc type from content or filename
        doc_type_patterns = {
            "Policy": r"(policy|policies)",
            "SOP": r"(sop|standard operating procedure|procedure)",
            "Guide": r"(guide|guidelines|guideline)",
            "FAQ": r"(faq|frequently asked questions)"
        }
        
        content_lower = content.lower()
        for doc_type, pattern in doc_type_patterns.items():
            if re.search(pattern, content_lower) or re.search(pattern, filename.lower()):
                metadata["doc_type"] = doc_type
                break
        
        # Extract date from content (looking for Date: or Effective Date:)
        date_match = re.search(r"(Date|Effective Date):\s*(\d{4}-\d{2}-\d{2}|\w+\s+\d{1,2},?\s+\d{4})", content)
        if date_match:
            date_str = date_match.group(2)
            # Try to parse and convert to YYYY-MM-DD format
            try:
                # Handle "January 15, 2024" format
                if re.match(r"\w+\s+\d{1,2},?\s+\d{4}", date_str):
                    parsed_date = datetime.strptime(date_str.replace(',', ''), "%B %d %Y")
                    metadata["created_date"] = parsed_date.strftime("%Y-%m-%d")
                else:
                    metadata["created_date"] = date_str
            except:
                pass
        
        return metadata
    
    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split document into overlapping chunks"""
        content = document["content"]
        metadata = document["metadata"]
        source_file = document["source_file"]
        
        # Simple chunking by characters with overlap
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(content):
            end = start + self.chunk_size
            chunk_text = content[start:end]
            
            # Try to break at sentence boundary if possible
            if end < len(content):
                last_period = chunk_text.rfind('.')
                last_newline = chunk_text.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > self.chunk_size * 0.7:  # Only if we're not losing too much
                    chunk_text = chunk_text[:break_point + 1]
                    end = start + break_point + 1
            
            # Clean up chunk
            chunk_text = chunk_text.strip()
            
            if len(chunk_text) > 100:  # Only keep substantial chunks
                chunks.append({
                    "text": chunk_text,
                    "department": metadata["department"],
                    "doc_type": metadata["doc_type"],
                    "created_date": metadata["created_date"],
                    "source_file": source_file,
                    "chunk_id": chunk_id
                })
                chunk_id += 1
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            if start >= len(content):
                break
        
        return chunks
    
    def process_all_documents(self, data_dir: str = "data") -> List[Dict[str, Any]]:
        """Load and chunk all documents"""
        documents = self.load_documents(data_dir)
        
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        
        print(f"✅ Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks


# Test the document processor
if __name__ == "__main__":
    processor = DocumentProcessor(chunk_size=800, chunk_overlap=200)
    
    # Load and process documents
    chunks = processor.process_all_documents("data")
    
    # Display sample chunk
    if chunks:
        print("\n" + "="*60)
        print("SAMPLE CHUNK:")
        print("="*60)
        sample = chunks[0]
        print(f"Department: {sample['department']}")
        print(f"Doc Type: {sample['doc_type']}")
        print(f"Date: {sample['created_date']}")
        print(f"Source: {sample['source_file']}")
        print(f"Chunk ID: {sample['chunk_id']}")
        print(f"\nText Preview (first 200 chars):")
        print(sample['text'][:200] + "...")
        print("="*60)
        
        # Show distribution
        from collections import Counter
        dept_dist = Counter(c['department'] for c in chunks)
        type_dist = Counter(c['doc_type'] for c in chunks)
        
        print(f"\nDepartment Distribution: {dict(dept_dist)}")
        print(f"Doc Type Distribution: {dict(type_dist)}")