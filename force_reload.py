# Save as force_reload.py
import os
import shutil
from dotenv import load_dotenv
from rag_engine import RAGEngine

# Load environment variables
load_dotenv()

# Define paths
docs_dir = "static_docs"
db_dir = "chroma_db"

# Check if documents exist
if os.path.exists(docs_dir):
    files = os.listdir(docs_dir)
    pdf_count = len([f for f in files if f.endswith('.pdf')])
    txt_count = len([f for f in files if f.endswith('.txt')])
    md_count = len([f for f in files if f.endswith('.md')])
    
    print(f"Found {pdf_count} PDFs, {txt_count} text files, and {md_count} markdown files in {docs_dir}")
else:
    print(f"Directory {docs_dir} doesn't exist. Creating it...")
    os.makedirs(docs_dir)
    pdf_count = txt_count = md_count = 0

# Remove existing database if it exists
if os.path.exists(db_dir):
    print(f"Removing existing database at {db_dir}")
    shutil.rmtree(db_dir)
    print("Database removed")

# Create a test document if no documents exist
if pdf_count + txt_count + md_count == 0:
    print("No documents found. Creating a test document...")
    with open(f"{docs_dir}/test_document.txt", "w") as f:
        f.write("This is a test document about Harvard research. The Harvard paper discusses methodology and findings related to artificial intelligence applications. " 
                "The paper was written by researchers at Harvard University and examines the ethical implications of AI in healthcare and education. "
                "One of the main conclusions of the Harvard paper is that AI systems need more oversight and transparency.")
    print("Test document created")

# Initialize RAG engine and load documents
print("Initializing RAG engine and loading documents...")
rag_engine = RAGEngine(docs_dir=docs_dir, db_dir=db_dir)
rag_engine.load_documents()
print("Documents loaded")

# Verify documents were added to the database
try:
    from langchain_chroma import Chroma
except ImportError:
    try:
        from langchain.vectorstores import Chroma
    except ImportError:
        from langchain_community.vectorstores import Chroma

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
try:
    vectordb = Chroma(persist_directory=db_dir, embedding_function=embeddings)
    doc_count = vectordb._collection.count()
    print(f"Database was initialized with approximately {doc_count} document chunks!")
    
    if doc_count > 0:
        print("\nTesting a query...")
        results = vectordb.similarity_search("Harvard paper", k=1)
        if results:
            print("Found relevant document:")
            print(f"Content: {results[0].page_content[:100]}...")
        else:
            print("No results found for test query.")
    
except Exception as e:
    print(f"ERROR checking database: {str(e)}")