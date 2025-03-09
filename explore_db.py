import os
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_chroma import Chroma
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the embedding function (same as in your application)
embeddings = OpenAIEmbeddings()

# Load the existing Chroma database
db_dir = "chroma_db"
vectordb = Chroma(persist_directory=db_dir, embedding_function=embeddings)

# Print basic info
print(f"Collection contains approximately {vectordb._collection.count()} documents")

# Get all the documents
docs = vectordb.get()
if docs and 'documents' in docs:
    print(f"Found {len(docs['documents'])} documents")
    
    # Print first 3 documents and their sources
    for i in range(min(3, len(docs['documents']))):
        doc = docs['documents'][i]
        metadata = docs['metadatas'][i] if 'metadatas' in docs else {}
        print(f"\nDocument {i+1}:")
        print(f"Source: {metadata.get('source', 'unknown')}")
        print(f"Content preview: {doc[:200]}...")

# Perform a similarity search
query = "What is the main topic?"
results = vectordb.similarity_search(query, k=3)
print("\nTop 3 most similar documents to the query 'What is the main topic?':")
for i, doc in enumerate(results):
    print(f"\nResult {i+1}:")
    print(f"Source: {doc.metadata.get('source', 'unknown')}")
    print(f"Content preview: {doc.page_content[:200]}...")