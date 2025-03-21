import streamlit as st
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pymongo import MongoClient
from dotenv import load_dotenv
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import TextLoader

# Configuration
load_dotenv()

# Initialize session state
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None
    
def init_components():
    # Set up embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    # MongoDB connection
    client = MongoClient(os.getenv("MONGO_URI"))
    collection = client[os.getenv("DB_NAME")][os.getenv("COLLECTION_NAME")]
    
    # Create vector store
    vector_store = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name=os.getenv("INDEX_NAME")
    )
    
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model="learnlm-1.5-pro-experimental",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.1
    )
    
    return vector_store, llm

def create_rag_chain(vector_store, llm):
    template = """{context}

---

Given the context above, answer the question as best as possible.

Question: {question}

Answer: """
    
    prompt = PromptTemplate.from_template(template)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# Streamlit UI
st.title("RAG Chat Interface")

# Initialize RAG chain if not already done
if st.session_state.rag_chain is None:
    with st.spinner("Initializing RAG system..."):
        vector_store, llm = init_components()
        st.session_state.rag_chain = create_rag_chain(vector_store, llm)
    st.success("RAG system initialized!")

# Chat interface
query = st.text_input("Ask a question:")
if st.button("Submit"):
    if query:
        with st.spinner("Generating response..."):
            response = st.session_state.rag_chain.invoke(query)
            st.write("Answer:", response)