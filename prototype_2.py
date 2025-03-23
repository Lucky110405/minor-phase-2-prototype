import os
import json
from dotenv import load_dotenv
import requests
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch

# Load environment variables
load_dotenv()

# Configuration
OPENROUTER_GEMMA_API_KEY = os.getenv("OPENROUTER_GEMMA_API_KEY")
BASE_URL="https://openrouter.ai/api/v1"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
INDEX_NAME = os.getenv("INDEX_NAME")

class OpenRouterLLM:
    def __init__(self, api_key, temperature=0.1):
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "LangChain Integration",
            "Content-Type": "application/json"
        }
        self.temperature = temperature

    def __call__(self, prompt):
        # Convert PromptValue to string if needed
        if hasattr(prompt, 'to_string'):
            prompt = prompt.to_string()
        
        try:
            print(f"Sending request to API...")
            response = requests.post(
                f"{BASE_URL}/chat/completions",
                headers=self.headers,
                json={
                    "model": "google/gemma-3-4b-it:free",
                    "messages": [{"role": "user", "content": str(prompt)}],
                    "temperature": self.temperature,
                    "max_tokens": 1000,
                    "stream": False
                }
            )

            # Print response for debugging
            print(f"API Response Status: {response.status_code}")
            
            # Add response validation
            if response.status_code != 200:
                print(f"API Error: {response.status_code} - {response.text}")
                return "Sorry, there was an error with the API request."
                
            response_json = response.json()
            
            # Add error handling for missing data
            if not response_json.get("choices"):
                print(f"Unexpected API response format: {response_json}")
                return "Sorry, received an unexpected response format."

            content = response_json["choices"][0]["message"]["content"]
            print(f"Response: {content}")   
            # Extract the message content safely
            
        except Exception as e:
            print(f"Error calling OpenRouter API: {e}")
            return "Sorry, I encountered an error processing your request."

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
    llm = OpenRouterLLM(api_key=OPENROUTER_GEMMA_API_KEY)
    return vector_store, llm

# Create default JSON file if user doesnt give any data
def create_default_json(file_path):
    """Create a default JSON file with empty structure"""
    default_data = {
        "status": "general_mode",
        "message": "No user data provided - using general advisory mode"
    }
    
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(default_data, f, indent=4)
        return default_data
    except Exception as e:
        print(f"Error creating default JSON: {e}")
        return None

# Read user data
def read_user_data(file_path):
    """Read user data from JSON file or create default"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            # If file exists but is empty, create default
            if not data:
                return create_default_json(file_path)
            return data
    except FileNotFoundError:
        print("No user data file found. Creating default template...")
        return create_default_json(file_path)
    except json.JSONDecodeError:
        print("Invalid JSON format. Creating new default template...")
        return create_default_json(file_path)
    except Exception as e:
        print(f"Error reading file: {e}. Using default template...")
        return create_default_json(file_path)

# RAG chain setup
def create_rag_chain(vector_store, llm, user_data):
    
    """Create chain with user data context"""
    template = """You are a professional financial advisor. Based on standard banking practices and financial prudence, provide a clear financial advice in simple plain text format.

Current Financial Information of client:
{user_data}

Context from Knowledge Base:
{context}

Question: {question}

Provide a detailed but concise response with clear recommendations in PLAIN TEXT ONLY:"""
    
    prompt = PromptTemplate.from_template(template)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    
    # Convert user_data to JSON string for template insertion
    user_data_str = json.dumps(user_data, indent=2)
    
    rag_chain = (
        {
            "context": retriever | format_docs, 
            "question": RunnablePassthrough(),
            "user_data": lambda _: user_data_str  # Add user_data as a constant
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# Format documents

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Main execution
if __name__ == "__main__":
    # Initialize components
    vector_store, llm = init_components()

    # Use absolute path for the file
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(BASE_DIR, "user_document", "test_faq.json")
    
    # This will check if file exists and create a default JSON if not found
    user_data = read_user_data(file_path)
    
    # Add a message to show if using default or actual user data
    if user_data.get("status") == "general_mode":
        print("Using general mode with default settings (no user data found)")
    else:
        print("Using personalized mode with user-provided data")
    
    # Create RAG chain    
    rag_chain = create_rag_chain(vector_store, llm, user_data)
    
    # Chat interface
    print("RAG Based Financial Advisory System Ready. Type 'exit' to quit.")
    
    while True:
        query = input("\nQuestion: ")
        if query.lower() == 'exit':
            break
        
        try:
            response = rag_chain.invoke(query)
            print(f"\nAdvice: {response}")
        except Exception as e:
            print(f"Error processing query: {e}")
            print("Please try another question.")


