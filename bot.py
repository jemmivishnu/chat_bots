import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv

# Load environment
load_dotenv()
# OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OpenAI API key is not set. Please check your `.env` file.")

# Streamlit app
st.set_page_config(page_title="Financial Report Analyzer", layout="wide")
st.title("Financial Report Analyzer with GPT-4 Turbo")

# File upload section
uploaded_file = st.file_uploader("Upload a Financial Report (PDF)", type="pdf")
if uploaded_file:
    st.write(f"Uploaded file: {uploaded_file.name}")

    # Load and process the uploaded PDF
    loader = PyPDFLoader(uploaded_file)
    pages = loader.load_and_split()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    split_documents = text_splitter.split_documents(pages)

    # Create embeddings for the documents
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    vectorstore.add_documents(split_documents)
    vectorstore.persist()

# Conversational Retrieval Chain with similarity search
if os.path.exists("chroma_db"):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-ada-002")
    vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})  # Retrieve top 3 similar documents

    llm = ChatOpenAI(model="gpt-4", openai_api_key=OPENAI_API_KEY)
    chain = ConversationalRetrievalChain.from_llm(llm, retriever)

    st.write("Conversational Retrieval Chain initialized.")

# Query section
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

query = st.text_input("Ask a question about the financial report:")
if query:
    # Perform similarity search on the vectorstore
    response = chain.run({"question": query, "chat_history": st.session_state["chat_history"]})

    # Append the query and response as a tuple to the chat_history
    st.session_state["chat_history"].append((query, response))

    # Display the response
    st.write("Response:")
    st.write(response)

    # Display chat history
    st.write("Chat History:")
    for entry in st.session_state["chat_history"]:
        st.write(f"**Q:** {entry[0]}")  # entry[0] is the question
        st.write(f"**A:** {entry[1]}")  # entry[1] is the answer
