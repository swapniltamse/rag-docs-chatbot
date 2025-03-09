import os
import glob
from typing import List, Dict, Any

from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
# from langchain.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma 
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
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.4)

        # Create directory for docs if it doesn't exist
        os.makedirs(docs_dir, exist_ok=True)

        # Initialize conversation history
        self.conversation_history = []
        self.max_history_length = 5  # Keep last 5 exchanges

    def add_to_history(self, user_message: str, assistant_response: str):
        """Add a message exchange to conversation history"""
        self.conversation_history.append({
            "user": user_message,
            "assistant": assistant_response
        })

        # Keep history to a manageable size
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]

    def format_history(self) -> str:
        """Format conversation history for inclusion in prompts"""
        if not self.conversation_history:
            return ""

        history_text = "Previous conversation:\n"
        for exchange in self.conversation_history:
            history_text += f"User: {exchange['user']}\n"
            history_text += f"Assistant: {exchange['assistant']}\n"

        return history_text

    def generate_response_with_history(self, query: str) -> dict:
        """Generate a response using RAG with conversation history"""
        # Load the vector store
        vectordb = Chroma(
            persist_directory=self.db_dir,
            embedding_function=self.embeddings
        )

        # Create a retriever
        retriever = vectordb.as_retriever(
            search_kwargs={"k": 5}  # Retrieve top 5 most relevant chunks
        )

        # Retrieve documents
        retrieved_docs = retriever.get_relevant_documents(query)

        # Track sources for citation
        source_documents = [doc.metadata.get('source', 'unknown') for doc in retrieved_docs]

        # Get formatted conversation history
        history = self.format_history()

        # Create a prompt template that includes conversation history
        prompt = ChatPromptTemplate.from_template("""
        You are a helpful, friendly assistant that helps users find information in their documents.

        {history}

        When answering the current question, follow these guidelines:
        1. Be conversational and warm in your tone
        2. If the answer is in the context, provide it clearly
        3. If the answer isn't fully in the context, acknowledge what is known and what isn't
        4. Always maintain an encouraging and helpful tone
        5. Use the conversation history for context but focus on answering the current question
        6. If you reference information from the documents, be accurate and don't fabricate details

        Context from documents:
        {context}

        Current Question: {input}

        Your helpful response:
        """)

        # Create a document chain
        document_chain = create_stuff_documents_chain(
            self.llm,
            prompt,
            document_variable_name="context"
        )

        # Create a retrieval chain
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Generate response
        response = retrieval_chain.invoke({
            "input": query,
            "history": history
        })

        # Add this exchange to history
        self.add_to_history(query, response["answer"])

        return {
            "answer": response["answer"],
            "sources": list(set(source_documents))  # Remove duplicates
        }
        
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
    

    def is_db_initialized(self) -> bool:
        """Check if vector database exists and has data"""
        return os.path.exists(self.db_dir) and len(os.listdir(self.db_dir)) > 0

    def generate_response_with_fallback(self, query: str) -> dict:
        """Generate a response with fallback for when no relevant documents are found"""
        # Load the vector store
        vectordb = Chroma(
            persist_directory=self.db_dir,
            embedding_function=self.embeddings
        )

        # Create a retriever
        retriever = vectordb.as_retriever(
            search_kwargs={"k": 5, "score_threshold": 0.7}  # Add score threshold to filter low relevance
        )

        # Retrieve documents
        retrieved_docs = retriever.get_relevant_documents(query)

        # Track sources for citation
        source_documents = [doc.metadata.get('source', 'unknown') for doc in retrieved_docs]

        if not retrieved_docs:
            # Create a general fallback prompt
            fallback_prompt = ChatPromptTemplate.from_template("""
            You are a helpful assistant. The user has asked: "{input}"

            You don't have specific information about this in your knowledge base. 

            Please:
            1. Acknowledge that you don't have specific information about this topic in your documents
            2. Offer a thoughtful, empathetic response that's still helpful
            3. Suggest related topics the user might want to ask about instead
            4. Ask a follow-up question to better understand what they're looking for

            Your response:
            """)

            # Set a slightly higher temperature for more creative fallback responses
            self.llm.temperature = 0.5

            # Generate fallback response
            fallback_chain = fallback_prompt | self.llm
            response = fallback_chain.invoke({"input": query})

            return {
                "answer": response.content,
                "sources": [],
                "fallback": True
            }

        # If we have relevant documents, proceed with normal RAG
        # Use a moderate temperature for factual but conversational responses
        self.llm.temperature = 0.3

        # Create a prompt template
        prompt = ChatPromptTemplate.from_template("""
        You are a helpful, friendly assistant that helps users find information in their documents.

        When answering questions, follow these guidelines:
        1. Be conversational and warm in your tone
        2. If the answer is in the context, provide it clearly
        3. If the answer isn't fully in the context, acknowledge what is known and what isn't
        4. Always maintain an encouraging and helpful tone
        5. If you reference information from the documents, be accurate and don't fabricate details

        Context:
        {context}

        User Question: {input}

        Your helpful response:
        """)

        # Create a document chain
        document_chain = create_stuff_documents_chain(self.llm, prompt)

        # Create a retrieval chain
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Generate response
        response = retrieval_chain.invoke({"input": query})

        return {
            "answer": response["answer"],
            "sources": list(set(source_documents)),  # Remove duplicates
            "fallback": False
        }
        
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
    
    def add_to_history(self, user_message: str, assistant_response: str):
        """Add a message exchange to conversation history"""
        self.conversation_history.append({
            "user": user_message,
            "assistant": assistant_response
        })
        
        # Keep history to a manageable size
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]

    def generate_response(self, query: str) -> str:
        """Simple debugging version of generate_response"""
        try:
            print(f"Processing query: {query}")

            # Load the vector store
            vectordb = Chroma(
                persist_directory=self.db_dir,
                embedding_function=self.embeddings
            )

            # Create a retriever
            retriever = vectordb.as_retriever(
                search_kwargs={"k": 5}  # Retrieve top 5 most relevant chunks
            )

            # Create a prompt template
            prompt = ChatPromptTemplate.from_template("""
            Answer the following question based on the provided context. 
            If you don't know the answer or if the answer isn't in the context, say so - don't make up information.

            Context:
            {context}

            Question: {input}

            Answer:
            """)

            # Create a document chain
            document_chain  = create_stuff_documents_chain(self.llm, prompt)

            # Create a retrieval chain
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            # Generate response
            response = retrieval_chain.invoke({"input": query})

            return response["answer"]
        except Exception as e:
            print(f"ERROR in generate_response: {str(e)}")
            import traceback
            traceback.print_exc()

            return f"I apologize, but I encountered an error: {str(e)}"