
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
OPENROUTER_DEEPSEEK_API_KEY = os.getenv("OPENROUTER_DEEPSEEK_API_KEY")
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
            response = requests.post(
                f"{BASE_URL}/chat/completions",  # Updated endpoint
                headers=self.headers,
                json={
                    "model": "deepseek/deepseek-r1-zero:free",
                    "messages": [{"role": "user", "content": str(prompt)}],
                    "temperature": self.temperature,
                    "max_tokens": 1000,  # Add max tokens parameter
                    "stream": False  # Ensure we get complete response
                }
            )

            # Print response for debugging
            print(f"API Response Status: {response.status_code}")
            print(f"API Response Headers: {response.headers}")
            
            # Add response validation
            if response.status_code != 200:
                print(f"API Error: {response.status_code} - {response.text}")
                return "Sorry, there was an error with the API request."
                
            response_json = response.json()
            
            # Add error handling for missing data
            if not response_json.get("choices"):
                print(f"Unexpected API response format: {response_json}")
                return "Sorry, received an unexpected response format."
                
            # Extract the message content safely
            try:
                content = response_json["choices"][0]["message"]["content"]
                return content.strip()
            except (KeyError, IndexError) as e:
                print(f"Error extracting content from response: {e}")
                print(f"Response structure: {response_json}")
                return "Sorry, couldn't extract the response content."
            
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
    llm = OpenRouterLLM(api_key=OPENROUTER_DEEPSEEK_API_KEY)
    return vector_store, llm

# Read user data
def read_user_data(file_path):
    """Read user data from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Invalid JSON format in file: {file_path}")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

# RAG chain setup
def create_rag_chain(vector_store, llm, user_data):
    
    """Create chain with user data context"""
    template = """You are a professional financial advisor. Based on standard banking practices and financial prudence, provide a clear financial advice in simple plain text format.
DO NOT use any special formatting, LaTeX commands, or symbols.

Current Financial Information of client:
{user_data}

Context from Knowledge Base:
{context}


Please analyze and provide advice considering:
1. Standard debt-to-income ratio (50% max)
2. Typical car loan terms (5-7 years)
3. Current market interest rates (8-12%)
4. Monthly repayment capacity
5. Financial prudence and risk assessment

Question: {question}

Provide a detailed but concise response with clear recommendations:"""
    
    prompt = PromptTemplate.from_template(template)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
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
    
    # Read user data
    user_data = read_user_data(file_path)
    
    # Create personalized RAG chain    
    rag_chain = create_rag_chain(vector_store, llm, user_data)
    
    # Chat interface
    print("RAG Based Financial Advisory System Ready. Type 'exit' to quit.")
    while True:
        query = input("\nQuestion: ")
        if query.lower() == 'exit':
            break
        
        response = rag_chain.invoke(query)
        print(f"\nAnswer: {response}")


