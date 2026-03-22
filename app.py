import streamlit as st
from retrieval import RAGRetrieval
import os
from datetime import datetime

# Page config
st.set_page_config(
    page_title="EnterpriseRAG",
    page_icon="🔍",
    layout="wide"
)

# Initialize RAG system
@st.cache_resource
def initialize_rag():
    """Initialize and load RAG system (cached)"""
    with st.spinner("🔄 Initializing RAG system..."):
        rag = RAGRetrieval()
        
        # Check if documents are already ingested
        try:
            stats = rag.qdrant.get_stats()
            if stats['total_points'] == 0:
                # Ingest documents if empty
                rag.ingest_documents("data")
        except:
            # First time setup
            rag.ingest_documents("data")
        
        return rag

rag = initialize_rag()

# Header
st.title("🔍 EnterpriseRAG - Knowledge Assistant")
st.markdown("*Ask questions about company policies, procedures, and guidelines*")

# Sidebar - Filters
st.sidebar.header("🎯 Search Settings")

# NEW: Query Decomposition Toggle
use_decomposition = st.sidebar.checkbox(
    "🧠 Smart Query Decomposition",
    value=False,
    help="Automatically break down complex questions into sub-queries for better answers"
)

if use_decomposition:
    st.sidebar.info("💡 Complex questions will be automatically analyzed and split into focused sub-queries")
else:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Manual Filters")
    
    # Department filter
    department_options = ["All Departments", "Engineering", "Hr"]
    selected_dept = st.sidebar.selectbox(
        "Department",
        department_options,
        help="Filter by department"
    )

    # Doc type filter
    doc_type_options = ["All Types", "Policy", "SOP", "Guide", "FAQ"]
    selected_type = st.sidebar.selectbox(
        "Document Type",
        doc_type_options,
        help="Filter by document type"
    )

    # Date range filter
    st.sidebar.subheader("📅 Date Range")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        date_from = st.date_input(
            "From",
            value=None,
            help="Filter documents from this date"
        )
    with col2:
        date_to = st.date_input(
            "To",
            value=None,
            help="Filter documents until this date"
        )

# Advanced settings
with st.sidebar.expander("⚙️ Advanced Settings"):
    top_k = st.slider("Number of sources", 1, 10, 5)
    show_scores = st.checkbox("Show relevance scores", value=True)
    if use_decomposition:
        show_decomposition = st.checkbox("Show decomposition details", value=True)

# Stats in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Database Stats")
stats = rag.qdrant.get_stats()
st.sidebar.metric("Total Documents Indexed", stats['total_points'])

# Main area - Query interface
st.header("💬 Ask a Question")

# Sample questions
with st.expander("📝 Sample Questions"):
    if use_decomposition:
        st.markdown("""
        **Complex questions (perfect for decomposition):**
        - What is the deployment process for remote engineering employees?
        - How do engineering and HR teams handle onboarding?
        - What are the code review and expense reimbursement procedures?
        - What policies apply to remote work and PTO for contractors?
        """)
    else:
        st.markdown("""
        **Simple questions:**
        - What is the remote work policy?
        - What are the deployment procedures?
        - How do I request time off?
        - What are the code review guidelines?
        - What is the expense reimbursement process?
        """)

# Query input
query = st.text_area(
    "Your Question:",
    placeholder="E.g., What is the deployment process for remote engineering employees?",
    height=100
)

# Search button
search_button = st.button("🔍 Search", type="primary", use_container_width=True)

if search_button and query.strip():
    
    if use_decomposition:
        # Use query decomposition
        with st.spinner("🧠 Analyzing and decomposing query..."):
            result = rag.query_with_decomposition(
                question=query,
                top_k=top_k
            )
        
        # Display results
        st.markdown("---")
        
        # Show decomposition details
        if show_decomposition:
            with st.expander("🔍 Query Decomposition Details", expanded=True):
                st.write(f"**Needs Decomposition:** {result['decomposition']['needs_decomposition']}")
                st.write(f"**Number of Sub-queries:** {len(result['sub_results'])}")
                
                for idx, sub in enumerate(result['sub_results'], 1):
                    st.markdown(f"**Sub-query {idx}:** {sub['sub_query']}")
                    st.caption(f"Department: {sub['department'] or 'All'} | Doc Type: {sub['doc_type'] or 'All'}")
                    st.caption(f"Sources found: {len(sub['sources'])}")
                    st.markdown("---")
        
        # Answer
        st.subheader("💡 Synthesized Answer")
        st.markdown(result["answer"])
        
    else:
        # Standard query with manual filters
        filters = {}
        
        if selected_dept != "All Departments":
            filters["department"] = selected_dept
        
        if selected_type != "All Types":
            filters["doc_type"] = selected_type
        
        if date_from:
            filters["date_from"] = date_from.strftime("%Y-%m-%d")
        
        if date_to:
            filters["date_to"] = date_to.strftime("%Y-%m-%d")
        
        # Execute query
        with st.spinner("🤔 Searching knowledge base..."):
            result = rag.query(
                question=query,
                **filters,
                top_k=top_k
            )
        
        # Display results
        st.markdown("---")
        
        # Answer
        st.subheader("💡 Answer")
        st.markdown(result["answer"])
        
        # Filters applied
        if result["filters_applied"]:
            st.info(f"**Filters applied:** {', '.join([f'{k}: {v}' for k, v in result['filters_applied'].items()])}")
    
    # Sources (common for both modes)
    st.markdown("---")
    st.subheader(f"📚 Sources ({len(result['sources'])} documents)")
    
    if result["sources"]:
        for idx, source in enumerate(result["sources"], 1):
            with st.expander(f"📄 {idx}. {source['source_file']} - {source['department']} ({source['doc_type']})"):
                if show_scores:
                    st.caption(f"**Relevance Score:** {source['score']:.3f}")
                st.markdown(f"**Department:** {source['department']}")
                st.markdown(f"**Type:** {source['doc_type']}")
                st.markdown("**Content:**")
                st.text_area(
                    "Source text",
                    source['text'],
                    height=200,
                    key=f"source_{idx}",
                    label_visibility="collapsed"
                )
    else:
        st.warning("No relevant sources found.")

elif search_button and not query.strip():
    st.warning("⚠️ Please enter a question.")

# Footer
st.markdown("---")
col1, col2 = st.columns([3, 1])
with col1:
    st.caption("🤖 Powered by Ollama (Mistral 7B + nomic-embed-text) | 💾 Qdrant Vector Database")
with col2:
    if use_decomposition:
        st.caption("🧠 Smart Mode: ON")
    else:
        st.caption("🎯 Manual Mode")