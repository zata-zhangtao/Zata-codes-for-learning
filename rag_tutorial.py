"""
RAG (Retrieval-Augmented Generation) Tutorial using LangChain
This tutorial demonstrates how to build a RAG application step by step.
"""

import os
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class RAGTutorial:
    def __init__(self):
        """Initialize the RAG tutorial with necessary components."""
        # Initialize the text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        # Initialize the embeddings model using DashScope
        self.embeddings = DashScopeEmbeddings(
            model="text-embedding-v2",
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
        )
        
        # Initialize the LLM using DashScope
        self.llm = ChatOpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model="qwen-plus"
        )

    def load_documents(self, file_path: str) -> List:
        """
        Load and split documents from a file.
        
        Args:
            file_path (str): Path to the document file
            
        Returns:
            List: List of document chunks
        """
        # Load the document
        loader = TextLoader(file_path)
        documents = loader.load()
        
        # Split the documents into chunks
        chunks = self.text_splitter.split_documents(documents)
        return chunks

    def create_vector_store(self, chunks: List) -> FAISS:
        """
        Create a vector store from document chunks.
        
        Args:
            chunks (List): List of document chunks
            
        Returns:
            FAISS: Vector store containing document embeddings
        """
        # Create and return the vector store
        vector_store = FAISS.from_documents(chunks, self.embeddings)
        return vector_store

    def create_qa_chain(self, vector_store: FAISS) -> RetrievalQA:
        """
        Create a question-answering chain.
        
        Args:
            vector_store (FAISS): Vector store containing document embeddings
            
        Returns:
            RetrievalQA: Question-answering chain
        """
        # Create the retriever
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        # Create and return the QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        return qa_chain

    def query(self, qa_chain: RetrievalQA, question: str) -> dict:
        """
        Query the RAG system with a question.
        
        Args:
            qa_chain (RetrievalQA): Question-answering chain
            question (str): Question to ask
            
        Returns:
            dict: Answer and source documents
        """
        return qa_chain({"query": question})

def main():
    """Main function to demonstrate the RAG system."""
    # Initialize the RAG tutorial
    rag = RAGTutorial()
    
    # Load and process documents
    print("Loading and processing documents...")
    chunks = rag.load_documents("example_doc.txt")
    
    # Create vector store
    print("Creating vector store...")
    vector_store = rag.create_vector_store(chunks)
    
    # Create QA chain
    print("Creating QA chain...")
    qa_chain = rag.create_qa_chain(vector_store)
    
    # Example questions
    questions = [
        "What is the main topic of the document?",
        "Can you summarize the key points?",
        "What are the important details mentioned?"
    ]
    
    # Query the system
    print("\nQuerying the RAG system:")
    for question in questions:
        print(f"\nQuestion: {question}")
        result = rag.query(qa_chain, question)
        print(f"Answer: {result['result']}")
        print("\nSource Documents:")
        for doc in result['source_documents']:
            print(f"- {doc.page_content[:200]}...")

if __name__ == "__main__":
    main() 