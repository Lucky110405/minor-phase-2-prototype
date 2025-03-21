import os
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pymongo import MongoClient
from dotenv import load_dotenv
from langchain_mongodb import MongoDBAtlasVectorSearch

# Configuration
load_dotenv()  # Loads variables from .env into the environment

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
INDEX_NAME = os.getenv("INDEX_NAME")

# Initialize components
def init_components():
    # Set up embeddings
   
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GOOGLE_API_KEY
    )
    
    # MongoDB connection
    client = MongoClient(MONGO_URI)
    collection = client[DB_NAME][COLLECTION_NAME]
    
    # Create vector store
    vector_store = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name=INDEX_NAME
    )
    
    # Initialize LLM
   
    llm = ChatGoogleGenerativeAI(
        model="learnlm-1.5-pro-experimental",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.1
    )
    
    return vector_store, llm

# Document ingestion
def ingest_documents(vector_store, file_path):
    from langchain_community.document_loaders import TextLoader
    
    loader = TextLoader(file_path)
    documents = loader.load()
    vector_store.add_documents(documents)
    print(f"Ingested {len(documents)} documents")

# RAG chain setup
def create_rag_chain(vector_store, llm):
    # Prompt template from JSON
    template = """{context}


---

Given the context above, answer the question as best as possible.

Question: {question}

Answer: """
    
    prompt = PromptTemplate.from_template(template)
    
    # Retriever setup
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    
    # Chain construction
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Main execution
if __name__ == "__main__":
    # Initialize components
    vector_store, llm = init_components()
    
    # Ingest documents (replace with your file path)
    file_path = "user_document/test_finance_report .txt"
    ingest_documents(vector_store, file_path)
    
    # Create RAG chain
    rag_chain = create_rag_chain(vector_store, llm)
    
    # Chat interface
    print("RAG System Ready. Type 'exit' to quit.")
    while True:
        query = input("\nQuestion: ")
        if query.lower() == 'exit':
            break
        
        response = rag_chain.invoke(query)
        print(f"\nAnswer: {response}")