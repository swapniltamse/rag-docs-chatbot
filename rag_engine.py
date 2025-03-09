import os
import glob
from typing import List, Dict, Any

from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import pypdf


class RAGEngine:
    def __init__(self, docs_dir: str, db_dir: str):
        """Initialize the RAG engine

        Args:
            docs_dir: Directory containing documents to index
            db_dir: Directory to store the vector database
        """
        self.docs_dir = docs_dir
        self.db_dir = db_dir
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)

        # Create directory for docs if it doesn't exist
        os.makedirs(docs_dir, exist_ok=True)

    def is_db_initialized(self) -> bool:
        """Check if vector database exists and has data"""
        return os.path.exists(self.db_dir) and len(os.listdir(self.db_dir)) > 0

    def load_documents(self) -> None:
        """Load and index documents from the docs directory"""
        print("Loading documents...")
        documents = []

        # Process PDF files
        for pdf_path in glob.glob(f"{self.docs_dir}/**/*.pdf", recursive=True):
            print(f"Processing {pdf_path}")
            documents.extend(self._extract_pdf_text(pdf_path))

        # Process text files
        for txt_path in glob.glob(f"{self.docs_dir}/**/*.txt", recursive=True):
            print(f"Processing {txt_path}")
            with open(txt_path, 'r', encoding='utf-8') as f:
                text = f.read()
                documents.append({"text": text, "source": txt_path})

        # Process markdown files
        for md_path in glob.glob(f"{self.docs_dir}/**/*.md", recursive=True):
            print(f"Processing {md_path}")
            with open(md_path, 'r', encoding='utf-8') as f:
                text = f.read()
                documents.append({"text": text, "source": md_path})

        if not documents:
            print("No documents found to process")
            return

        # Chunk the documents
        chunks = self._chunk_documents(documents)

        # Create or update the vector store
        self._create_vector_store(chunks)

        print(f"Processed {len(documents)} documents into {len(chunks)} chunks")

    def _extract_pdf_text(self, pdf_path: str) -> List[Dict[str, str]]:
        """Extract text from PDF files"""
        documents = []
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                for i, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text.strip():
                        documents.append({
                            "text": text,
                            "source": f"{pdf_path}:page{i+1}"
                        })
        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {e}")

        return documents

    def _chunk_documents(self, documents: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Split documents into chunks for embedding"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

        chunks = []
        for doc in documents:
            for chunk in text_splitter.split_text(doc["text"]):
                chunks.append({
                    "text": chunk,
                    "source": doc["source"]
                })

        return chunks

    def _create_vector_store(self, chunks: List[Dict[str, Any]]) -> None:
        """Create or update the vector store with document chunks"""
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [{"source": chunk["source"]} for chunk in chunks]

        # Create new vector store
        Chroma.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas,
            persist_directory=self.db_dir
        )

    def generate_response(self, query: str) -> str:
        """Generate a response using RAG"""
        # Load the vector store
        vectordb = Chroma(
            persist_directory=self.db_dir,
            embedding_function=self.embeddings
        )

        # Create a retriever
        retriever = vectordb.as_retriever(
            search_kwargs={"k": 5}  # Retrieve top 5 most relevant chunks
        )

        # Retrieve documents and log them for debugging
        retrieved_docs = retriever.get_relevant_documents(query)
        print(f"Retrieved {len(retrieved_docs)} documents")

        source_documents = []
        for i, doc in enumerate(retrieved_docs):
            print(f"Doc {i+1}: {doc.page_content[:100]}... (Source: {doc.metadata.get('source', 'unknown')})")
            source_documents.append(doc.metadata.get('source', 'unknown'))

        # A empathetic prompt template
        prompt = ChatPromptTemplate.from_template("""
    You are a helpful, friendly assistant that helps users find information in their documents.

    When answering questions, follow these guidelines:
    1. Be conversational and warm in your tone.
    2. If the answer is in the context, provide it clearly and concisely.
    3. If the answer isn't fully in the context, acknowledge what is known and what isn't.
    4. If the context doesn't contain relevant information, acknowledge the user's question positively
        before explaining that you don't have that specific information. Suggest related topics you 
        could help with instead.
    5. Always maintain an encouraging and helpful tone even when you can't provide a complete answer.

        Context:
    {context}

    User Question: {input}

    Your helpful response:
    """)

        # Set a balanced temperature (0.4) - factual but conversational
        self.llm.temperature = 0.4

        # Create a document chain
        document_chain = create_stuff_documents_chain(self.llm, prompt)

        # Create a retrieval chain
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Generate response
        response = retrieval_chain.invoke({"input": query})

        # Return both the answer and sources
        return {
            "answer": response["answer"],
            "sources": list(set(source_documents))
        }