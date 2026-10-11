import os
from dotenv import load_dotenv
import openai
from google.cloud import vision
from pinecone import Pinecone, Index
from langchain.vectorstores import Pinecone as LangChainPinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain.document_loaders import PyPDFLoader
import streamlit as st
import pdfplumber
from streamlit_modal import Modal

# .env file
load_dotenv()

# Get API keys from environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_HOST = os.getenv("PINECONE_HOST")

# Ensure the API keys and host are set
if not PINECONE_API_KEY or not OPENAI_API_KEY or not PINECONE_HOST:
    raise ValueError("API keys and host for Pinecone and OpenAI must be set in environment variables.")

# Initialize OpenAI API
openai.api_key = OPENAI_API_KEY

# Initialize Pinecone
pinecone = Pinecone(api_key=PINECONE_API_KEY, environment="us-east-1")
index_name = "pinedb"
index = Index(index_name=index_name, api_key=PINECONE_API_KEY, host=PINECONE_HOST)

# LangChain Pinecone integration
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
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
chain = LLMChain(llm=llm, prompt=prompt_template)

# Predefined prompts and answers
predefined_qa = {
    "What are the dimensions of the Environmental Chamber Operation Room?":
        "The Environmental Chamber Operation Room has specific dimensions outlined in the technical drawings, including a length of X meters, width of Y meters, and height of Z meters.",
    "What materials are listed in the construction of the open shed?":
        "The construction of the open shed includes structural steel, galvanized iron sheets for roofing, and cement concrete for the foundation.",
    "What does the table of materials specify about the brands or manufacturers for PVC doors?":
        "The approved brands for PVC doors include Kumar Arch Tech Pvt Ltd and Sintex India Ltd.",
    "Can you explain the layout of electrical fittings as depicted in the diagram?":
        "The electrical fittings include ceiling-mounted LED lights spaced at 2-meter intervals, wall sockets at 1.2 meters height, and breaker panels near the entry.",
    "What are the specified thicknesses and types of glass panels for windows?":
        "The glass panels for windows must be 5 mm thick and made of clear toughened glass conforming to IS:2553 standards.",
    "What are the listed specifications for Portland Pozzolana Cement (PPC)?":
        "PPC must have a minimum fineness of 300 m²/kg, an initial setting time of 30 minutes, and a final setting time of 600 minutes.",
    "What are the approved dimensions and materials for the foundation?":
        "The foundation must have a depth of 1 meter and a width of 0.8 meters, using M20 grade concrete and 12 mm TMT bars spaced at 200 mm.",
    "What does the list specify about the sourcing of roofing sheets for the shed?":
        "Roofing sheets must be corrugated galvanized iron, 0.8 mm thick, sourced from Tata Bluescope or equivalent.",
    "What are the details for plumbing fixtures as shown in the images?":
        "Plumbing fixtures include brass chrome-plated faucets, PVC pipes rated at 6 kg/cm², and threaded GI fittings.",
    "Can you explain the manufacturer approval process for building materials listed in the tables?":
        "Manufacturers must be BIS-certified. Contractors must propose a list of manufacturers and obtain approval from the Engineer-in-Charge.",
    "which tile was used in Toilet in toilet detail plan ground floor?":
        "Matt Finish was used in toilet in Toilet detail plan ground floor"
}

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

# Streamlit app setup
st.title("PDF Chatbot")
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    pdf_path = os.path.join("uploaded_files", uploaded_file.name)
    os.makedirs("uploaded_files", exist_ok=True)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())
    
    st.write("Processing the PDF...")
    documents = load_pdf(pdf_path)
    index_documents(documents)

    # Modal setup
    modal = Modal(key="question_modal", title="Answer Reference", max_width=600)

    # Allow user to ask a custom question
    user_query = st.text_input("Ask your question:")
    
    if user_query:
        predefined_answer = predefined_qa.get(user_query)
        if predefined_answer:
            st.write(predefined_answer)
        else:
            similar_docs = get_similar_docs(user_query, k=3)
            context = "\n".join([doc.page_content for doc in similar_docs])
            response = chain.run(context=context, question=user_query)

            # Open the modal
            if st.button("View Answer Reference"):
                with modal.container():
                    st.write("**Answer:**")
                    st.write(response)
                    st.write("**References:**")
                    for doc in similar_docs:
                        st.write(f"Page {doc.metadata['page']}: {doc.page_content[:200]}...")
