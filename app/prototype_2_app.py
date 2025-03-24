import streamlit as st
import os
import json
import sys
import time
from pathlib import Path

# Add the project root to the Python path
PROJECT_ROOT = str(Path(__file__).parent.parent.absolute())
sys.path.append(PROJECT_ROOT)

# Import components from prototype_2
from prototype_2 import (
    OpenRouterLLM, 
    init_components,
    read_user_data,
    create_rag_chain,
    test_retriever
)

# Page configuration
st.set_page_config(
    page_title="Financial Advisor AI",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.2rem;
        color: #64748B;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        position: relative;
    }
    .user-message {
        background-color: #101010;
        border-left: 4px solid #2563EB;

    }
    .assistant-message {
        background-color: #101010;
        border-left: 4px solid #2563EB;
    }
    .message-content {
        margin: 0;
    }
    .loading-spinner {
        text-align: center;
        margin: 2rem 0;
    }
    .sidebar-info {
        background-color: #101010;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">Financial Advisor AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Get personalized financial advice powered by RAG technology</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bank-building.png", width=80)
    st.title("System Information")

    # Initialize session state
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
        st.session_state.messages = []

    # Initialize RAG components
    if not st.session_state.initialized:
        with st.spinner("Initializing system..."):
            try:
                # Initialize components
                vector_store, llm = init_components()
                
                # Get user data
                file_path = Path(PROJECT_ROOT) / "user_document" / "test_faq.json"

                # This will check if file exists and create a default JSON if not found
                user_data = read_user_data(file_path)
                
                # Create RAG chain    
                rag_chain = create_rag_chain(vector_store, llm, user_data)
                
                # Store in session state
                st.session_state.vector_store = vector_store
                st.session_state.llm = llm
                st.session_state.user_data = user_data
                st.session_state.rag_chain = rag_chain
                st.session_state.initialized = True
                
                st.success("System initialized successfully!")
            except Exception as e:
                st.error(f"Error initializing system: {e}")

    # Display user data status
    if st.session_state.initialized:
        user_data = st.session_state.user_data
        if user_data.get("status") == "general_mode":
            st.markdown("""
            <div class="sidebar-info">
                <h3>General Advisory Mode</h3>
                <p>Using general financial knowledge without personalized data.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="sidebar-info">
                <h3>Personalized Advisory Mode</h3>
                <p>Using your personal financial data for tailored advice.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display some user data if available
            with st.expander("Your Financial Profile", expanded=False):
                st.json(user_data)

    # Tool section
    st.subheader("Tools")
    with st.expander("Test Vector Database", expanded=False):
        test_query = st.text_input("Test query:", "mutual funds")
        if st.button("Run Vector Search Test"):
            if st.session_state.initialized:
                with st.spinner("Searching database..."):
                    docs = test_retriever(st.session_state.vector_store, test_query)
                    st.success(f"Found {len(docs)} relevant documents")
                    
                    # Show results in expandable sections
                    for i, doc in enumerate(docs):
                        with st.expander(f"Document {i+1}"):
                            st.text(doc.page_content[:300] + "...")

    st.markdown("---")
    st.caption("© 2025 Financial Advisor AI")

# Chat interface
st.subheader("Ask your financial questions")

# Display chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []

for i, message in enumerate(st.session_state.messages):
    if message["role"] == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <p class="message-content"><strong>You:</strong> {message["content"]}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <p class="message-content"><strong>Financial Advisor:</strong> {message["content"]}</p>
        </div>
        """, unsafe_allow_html=True)

# Get user input
user_query = st.chat_input("Ask a financial question...", disabled=not st.session_state.initialized)

if user_query:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Display user message
    st.markdown(f"""
    <div class="chat-message user-message">
        <p class="message-content"><strong>You:</strong> {user_query}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get response
    if st.session_state.initialized:
        with st.spinner("Getting expert advice..."):
            try:
                # Get response from RAG chain
                response = st.session_state.rag_chain.invoke(user_query)
                
                # Add response to history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Display response with typewriter effect
                message_placeholder = st.empty()
                
                # Display full message after "typing" effect
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <p class="message-content"><strong>Financial Advisor:</strong> {response}</p>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.warning("System is not initialized yet. Please wait.")

# Help section at the bottom
with st.expander("About Financial Advisor AI"):
    st.markdown("""
    ### How to use this system
    
    This Financial Advisor AI uses Retrieval Augmented Generation (RAG) to provide accurate financial advice based on:
    
    1. **Knowledge Base**: Contains information about financial products, regulations, and best practices.
    2. **Your Financial Data**: If provided, enables personalized advice tailored to your situation.
    
    ### Sample Questions
    
    Try asking questions like:
    - "What are the best investment options for a short-term goal?"
    - "How do I build an emergency fund?"
    - "What's the typical interest rate for a home loan?"
    - "How much should I save for retirement?"
    - "Can I take a loan for a car of 20 lakhs if my income is 1 lakh per annum?"
    """)

# Export/Save chat
if st.sidebar.button("Export Chat History"):
    chat_export = ""
    for message in st.session_state.messages:
        role = "You" if message["role"] == "user" else "Financial Advisor"
        chat_export += f"{role}: {message['content']}\n\n"
    
    st.sidebar.download_button(
        label="Download Chat",
        data=chat_export,
        file_name="financial_advice_chat.txt",
        mime="text/plain"
    )

