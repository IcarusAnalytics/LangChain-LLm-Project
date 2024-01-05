from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from InstructorEmbedding import INSTRUCTOR


# Create a funtion that collects all text in pdfs submitted by user
def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text



# Create a function that divides the combined text into chunks
def get_text_chunks(raw_text):
    text_splitter=CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len)
    chunks = text_splitter.split_text(text)
    return chunks
     


# Create a function that embeds the text chunks into vectors and stores them in database
def get_vector_store(text_chunks):
    embeddings=INSTRUCTOR('hkunlp/instructor-xl')
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore
    



## Building the UI using streamlit
def main():
    load_dotenv()
    st.set_page_config(page_title="Chat With PDFs", page_icon=":books:")
    st.header("Chat With PDFs :books:")
    st.text_input("Ask a question about your PDF")
    with st.sidebar:

        st.subheader("Your Documents")
        pdf_docs = st.file_uploader(
            "Upload Your PDFs Here and click 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # Get pdf text
                raw_text= get_pdf_text(pdf_docs)

                #Get text chunks
                text_chunks= get_text_chunks(raw_text)


                # Create a vector store
                vector_store=get_vector_store(text_chunks)
           






if __name__=="__main__":
    main()