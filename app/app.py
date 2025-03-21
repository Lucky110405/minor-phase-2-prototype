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
from typing import List, Tuple

# Load environment variables
load_dotenv()

# Initialize session state
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history: List[Tuple[str, str]] = []
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

def init_components():
    """Initialize vector store and LLM components"""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        client = MongoClient(os.getenv("MONGO_URI"))
        collection = client[os.getenv("DB_NAME")][os.getenv("COLLECTION_NAME")]
        
        vector_store = MongoDBAtlasVectorSearch(
            collection=collection,
            embedding=embeddings,
            index_name=os.getenv("INDEX_NAME")
        )
        
        llm = ChatGoogleGenerativeAI(
            model="learnlm-1.5-pro-experimental",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.1
        )
        
        return vector_store, llm
    except Exception as e:
        st.error(f"Initialization failed: {str(e)}")
        return None, None

def create_rag_chain(vector_store, llm):
    """Create RAG processing chain"""
    template = """{context}

---
Given the context above, answer the question as best as possible.
Question: {question}
Answer: """
    
    prompt = PromptTemplate.from_template(template)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

def process_uploaded_files(vector_store):
    """Process and ingest uploaded files"""
    for uploaded_file in st.session_state.uploaded_files:
        if uploaded_file.name.endswith('.txt'):
            with st.spinner(f"Processing {uploaded_file.name}..."):
                try:
                    # Save uploaded file temporarily
                    with open(uploaded_file.name, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Load and ingest documents
                    loader = TextLoader(uploaded_file.name)
                    documents = loader.load()
                    vector_store.add_documents(documents)
                    st.success(f"Ingested {len(documents)} documents from {uploaded_file.name}")
                    
                    # Clean up temporary file
                    os.remove(uploaded_file.name)
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")

# Streamlit UI Configuration
st.set_page_config(
    page_title="RAG-Based Personalized Financial Advisory System",
    page_icon="🤖",
    layout="wide"
)

# Sidebar for document upload
with st.sidebar:
    st.header("Configuration")
    st.subheader("Document Management")
    
    uploaded_files = st.file_uploader(
        "Upload documents (TXT)",
        type=["txt"],
        accept_multiple_files=True,
        key="file_uploader"
    )
    
    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
    
    if st.button("Initialize/Update RAG System"):
        with st.status("Initializing system...", expanded=True) as status:
            st.write("Connecting to database...")
            vector_store, llm = init_components()
            
            if vector_store and llm:
                st.write("Processing documents...")
                process_uploaded_files(vector_store)
                
                st.write("Creating RAG chain...")
                st.session_state.rag_chain = create_rag_chain(vector_store, llm)
                status.update(label="System ready!", state="complete")
                st.session_state.chat_history = []
            else:
                status.update(label="Initialization failed", state="error")

# Main chat interface
st.title("RAG-Based Personalized Financial Advisory System")
st.markdown("""
    <style>
        .chat-message {
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            display: flex;
        }
        .user-message {
            background-color: #f0f2f6;
            margin-left: 20%;
        }
        .assistant-message {
            background-color: #e3f2fd;
            margin-right: 20%;
        }
    </style>
""", unsafe_allow_html=True)

# Chat history display
for role, message in st.session_state.chat_history:
    with st.chat_message(name=role):
        st.markdown(message)

# Chat input
if prompt := st.chat_input("Ask your question..."):
    if not st.session_state.rag_chain:
        st.error("Please initialize the RAG system first by uploading documents and clicking 'Initialize System'")
    else:
        # Add user question to chat history
        st.session_state.chat_history.append(("user", prompt))
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing documents..."):
                try:
                    response = st.session_state.rag_chain.invoke(prompt)
                    st.markdown(response)
                    st.session_state.chat_history.append(("assistant", response))
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")