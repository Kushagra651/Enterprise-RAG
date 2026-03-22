import ollama
from typing import List, Dict, Any, Optional
import json


class QueryDecomposer:
    def __init__(self, llm_model: str = "mistral:7b"):
        self.llm_model = llm_model
    
    def decompose_query(self, query: str) -> List[Dict[str, Any]]:
        """
        Break down complex query into sub-queries with routing info
        
        Returns: List of sub-queries with metadata
        [
            {
                "sub_query": "deployment process steps",
                "department": "Engineering",
                "doc_type": "SOP",
                "reasoning": "Technical procedure question"
            },
            ...
        ]
        """
        
        prompt = f"""You are a query decomposition expert for an enterprise knowledge base.

The knowledge base contains documents from two departments:
- Engineering: Technical guides, SOPs, development procedures, deployment processes, code review guidelines
- Hr: HR policies, leave policies, remote work guidelines, onboarding processes, benefits information

Document types available: Policy, SOP, Guide, FAQ

Your task: Analyze the user's question and determine if it needs to be broken down into sub-queries.

Rules:
1. If the question is simple and targets ONE topic → Return it as a single sub-query
2. If the question involves MULTIPLE topics or departments → Break it into 2-3 focused sub-queries
3. For each sub-query, specify which department and doc_type to search

User Question: "{query}"

Respond ONLY with valid JSON (no markdown, no explanations). Format:
{{
  "needs_decomposition": true/false,
  "sub_queries": [
    {{
      "sub_query": "the focused question",
      "department": "Engineering" or "Hr" or null,
      "doc_type": "Policy" or "SOP" or "Guide" or "FAQ" or null,
      "reasoning": "brief explanation"
    }}
  ]
}}

JSON Response:"""

        try:
            response = ollama.generate(
                model=self.llm_model,
                prompt=prompt,
                options={
                    "temperature": 0.1,
                    "num_predict": 500
                }
            )
            
            # Parse JSON response
            response_text = response['response'].strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            
            result = json.loads(response_text)
            
            return result
            
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse JSON: {e}")
            print(f"Raw response: {response_text}")
            # Fallback: treat as single query
            return {
                "needs_decomposition": False,
                "sub_queries": [
                    {
                        "sub_query": query,
                        "department": None,
                        "doc_type": None,
                        "reasoning": "Fallback - parsing failed"
                    }
                ]
            }
        except Exception as e:
            print(f"❌ Error in decomposition: {e}")
            return {
                "needs_decomposition": False,
                "sub_queries": [
                    {
                        "sub_query": query,
                        "department": None,
                        "doc_type": None,
                        "reasoning": "Fallback - error occurred"
                    }
                ]
            }
    
    def synthesize_answers(
        self,
        original_query: str,
        sub_results: List[Dict[str, Any]]
    ) -> str:
        """
        Combine answers from multiple sub-queries into coherent response
        """
        
        # Build context from sub-results
        context_parts = []
        for idx, result in enumerate(sub_results, 1):
            context_parts.append(f"""
Sub-question {idx}: {result['sub_query']}
Department: {result.get('department', 'All')}
Doc Type: {result.get('doc_type', 'All')}

Answer:
{result['answer']}

Sources:
{self._format_sources(result['sources'])}
""")
        
        full_context = "\n---\n".join(context_parts)
        
        prompt = f"""You are synthesizing information from multiple searches to answer a user's question.

Original Question: "{original_query}"

Information gathered from sub-queries:
{full_context}

Your task:
1. Combine the information into a single, coherent answer to the original question
2. Address all aspects of the user's question
3. Be concise but comprehensive
4. Mention relevant source documents when appropriate
5. If sub-answers conflict or have gaps, acknowledge this

Synthesized Answer:"""

        try:
            response = ollama.generate(
                model=self.llm_model,
                prompt=prompt,
                options={
                    "temperature": 0.3,
                    "num_predict": 600
                }
            )
            return response['response'].strip()
        except Exception as e:
            # Fallback: concatenate answers
            return "\n\n".join([r['answer'] for r in sub_results])
    
    def _format_sources(self, sources: List[Dict[str, Any]]) -> str:
        """Format sources for context"""
        if not sources:
            return "No sources found"
        
        formatted = []
        for src in sources[:3]:  # Top 3 sources
            formatted.append(f"- {src['source_file']} ({src['department']}, {src['doc_type']})")
        
        return "\n".join(formatted)


# Test the decomposer
if __name__ == "__main__":
    decomposer = QueryDecomposer()
    
    # Test cases
    test_queries = [
        "What is the remote work policy?",  # Simple - no decomposition
        "What is the deployment process for remote engineering employees?",  # Complex - needs decomposition
        "How do engineering and HR teams handle onboarding?",  # Multi-department
        "What are the code review and expense reimbursement procedures?"  # Multi-topic
    ]
    
    print("="*60)
    print("QUERY DECOMPOSITION TESTS")
    print("="*60)
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        result = decomposer.decompose_query(query)
        
        print(f"Needs Decomposition: {result['needs_decomposition']}")
        print(f"Sub-queries ({len(result['sub_queries'])}):")
        
        for idx, sq in enumerate(result['sub_queries'], 1):
            print(f"\n  {idx}. {sq['sub_query']}")
            print(f"     Department: {sq['department']}")
            print(f"     Doc Type: {sq['doc_type']}")
            print(f"     Reasoning: {sq['reasoning']}")
        
        print("-" * 60)