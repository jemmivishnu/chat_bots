import os
from dotenv import load_dotenv
import openai
from google.cloud import vision
from pinecone import Pinecone, Index
from langchain_pinecone import Pinecone as LangChainPinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.runnables import RunnableSequence
import streamlit as st
import pdfplumber



# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_HOST = os.getenv("PINECONE_HOST")

if not PINECONE_API_KEY or not OPENAI_API_KEY or not PINECONE_HOST:
    raise ValueError("API keys and host for Pinecone and OpenAI must be set in environment variables.")

# Initialize OpenAI
openai.api_key = OPENAI_API_KEY

# Initialize Pinecone
pinecone = Pinecone(api_key=PINECONE_API_KEY, environment="us-east-1")
index_name = "pinedb"
index = Index(index_name=index_name, api_key=PINECONE_API_KEY, host=PINECONE_HOST)

# -----------------------------
# LangChain Components
# -----------------------------
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
text_key = "text"
langchain_index = LangChainPinecone(index=index, embedding=embeddings, text_key=text_key)

# Prompt Template
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a helpful assistant. Based on the following context, answer the user's question.\n"
        "Context: {context}\n"
        "Question: {question}\n"
        "Answer:"
    ),
)

# LLM Initialization
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Modern chain syntax (RunnableSequence replaces deprecated LLMChain)
chain = prompt_template | llm

# -----------------------------
# Predefined Q&A
# -----------------------------
predefined_qa = {
    "What are the dimensions of the Environmental Chamber Operation Room?":
        "The Environmental Chamber Operation Room has specific dimensions outlined in the technical drawings, including a length of X meters, width of Y meters, and height of Z meters.",
}

# -----------------------------
# Helper Functions
# -----------------------------
def extract_text_with_pdfplumber(pdf_path):
    """Extract text from a PDF using pdfplumber."""
    with pdfplumber.open(pdf_path) as pdf:
        text_content = []
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                text_content.append(
                    Document(
                        page_content=text,
                        metadata={"source": pdf_path, "page": page_number}
                    )
                )
    return text_content

def load_pdf(file_path):
    """Load PDF, extract text and split into chunks."""
    text_content = extract_text_with_pdfplumber(file_path)
    text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    split_docs = text_splitter.split_documents(text_content)
    return split_docs

def index_documents(documents):
    """Index documents into Pinecone."""
    langchain_index.add_documents(documents)

def get_similar_docs(query, k=3):
    """Retrieve similar documents from Pinecone."""
    return langchain_index.similarity_search(query, k=k)

# -----------------------------
# Streamlit App UI
# -----------------------------
st.title("📘 PDF Chatbot")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    # Save uploaded file
    pdf_path = os.path.join("uploaded_files", uploaded_file.name)
    os.makedirs("uploaded_files", exist_ok=True)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())
    
    st.write("📄 Extracting and indexing PDF content...")
    documents = load_pdf(pdf_path)
    index_documents(documents)
    st.success("✅ PDF processed and indexed successfully!")

    # User query input
    user_query = st.text_input("Ask your question about the document:")

    if user_query:
        predefined_answer = predefined_qa.get(user_query)
        if predefined_answer:
            st.write(predefined_answer)
        else:
            st.write("🔍 Searching for relevant context...")
            similar_docs = get_similar_docs(user_query, k=3)
            context = "\n".join([doc.page_content for doc in similar_docs])
            
            st.write("💬 Generating answer...")
            response = chain.invoke({"context": context, "question": user_query})
            st.write(response.content)
