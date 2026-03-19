import os

dirs = [
    "app/api", "app/core", "app/services", "app/database",
    "data/raw", "data/processed", "data/sample_docs/engineering",
    "data/sample_docs/hr", "data/sample_docs/finance", "data/sample_docs/marketing",
    "tests", "evaluation", "scripts", "notebooks", "docker"
]

files = [
    "app/__init__.py", "app/main.py", "app/config.py",
    "app/api/__init__.py", "app/api/routes.py", "app/api/schemas.py",
    "app/core/__init__.py", "app/core/embeddings.py", "app/core/llm.py", "app/core/retrieval.py",
    "app/services/__init__.py", "app/services/document_processor.py", "app/services/metadata_extractor.py",
    "app/services/ingestion.py", "app/services/query_engine.py",
    "app/database/__init__.py", "app/database/qdrant_client.py", "app/database/schema.py",
    "tests/__init__.py", "tests/test_embeddings.py", "tests/test_retrieval.py", "tests/test_api.py",
    "evaluation/__init__.py", "evaluation/eval_queries.json", "evaluation/evaluate.py", "evaluation/metrics.py",
    "scripts/setup_qdrant.py", "scripts/ingest_sample_data.py", "scripts/generate_sample_docs.py",
    "notebooks/exploration.ipynb", "docker/Dockerfile", "docker/docker-compose.yml",
    ".env.example", ".gitignore", "requirements.txt", "README.md"
]

for d in dirs:
    os.makedirs(d, exist_ok=True)

for f in files:
    open(f, 'a').close()

print("✅ Project structure created successfully!")