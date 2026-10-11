from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
import streamlit as st
import os
import pdfplumber

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def extract_text_from_pdf(pdf_docs):
    """Extract text from PDF files."""
    text = ""
    for pdf in pdf_docs:
        with pdfplumber.open(pdf) as pdf_reader:
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
    return text

def create_vector_store(text_chunks):
    """Create a vector store with embeddings."""
    documents = [Document(page_content=chunk) for chunk in text_chunks]
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = Chroma.from_documents(
        documents, 
        embedding=embeddings,  # Note: No duplicate embedding_function
        persist_directory="chroma_db"
    )
    vectorstore.persist()

def main():
    st.title("Chat with Your PDF 📄")
    
    # Initialize chat history in session state if not already present
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    
    # Sidebar for uploading PDFs
    with st.sidebar:
        uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
        if st.button("Process"):
            if not uploaded_files:
                st.error("Please upload at least one PDF file.")
                return

            # Extract text
            raw_text = extract_text_from_pdf(uploaded_files)

            # Chunk the text
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
            text_chunks = text_splitter.split_text(raw_text)

            # Create vector store
            create_vector_store(text_chunks)
            st.success("PDF processed and embeddings created!")

    # Text input for querying
    query = st.text_input("Ask a question:")
    if query:
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vectorstore = Chroma(
            persist_directory="chroma_db", 
            embedding_function=embeddings
        )
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        
        llm = ChatOpenAI(model="gpt-4-turbo", openai_api_key=OPENAI_API_KEY)
        chain = ConversationalRetrievalChain.from_llm(llm, retriever)

        response = chain.run({"question": query, "chat_history": st.session_state["chat_history"]})
        
        # Update chat history with new question and answer
        st.session_state["chat_history"].append((query, response))

        # Display the response and chat history
        st.write(response)
        st.write("Chat History:")
        for entry in st.session_state["chat_history"]:
            st.write(f"**Q:** {entry[0]}")  # entry[0] is the question
            st.write(f"**A:** {entry[1]}")  # entry[1] is the answer


if __name__ == "__main__":
    main()
