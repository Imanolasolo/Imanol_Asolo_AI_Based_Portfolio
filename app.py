import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI 
from htmlTemplates import css, bot_template, user_template
import os

def get_pdf_text(pdf_list):
    text = ""
    for pdf in pdf_list:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    if not text_chunks:
        st.warning("Please upload the textual PDF file - this is PDF files of image")
        return None
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPEN_AI_APIKEY"])
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vector_store):
    llm = ChatOpenAI(openai_api_key=st.secrets["OPEN_AI_APIKEY"])
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userInput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, msg in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)

def main():
    load_dotenv()

    st.set_page_config(page_title="Imanol Asolo AI Agents handler", page_icon=":scroll:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "train" not in st.session_state:
        st.session_state.train = False

    if "pdf_text" not in st.session_state:
        st.session_state.pdf_text = ""

    st.header("Multi-Agents :books: - Chat handler :robot_face:")

    with st.sidebar:
        
        st.subheader(":books: Instructions")
        with st.expander("Expand instructions"):
            # Lista de instrucciones
            instrucciones = [
                "1. Choose betwen using Imanol Asolo bot or an agent created by yourself.",
                "2. Push Train agent button.",
                "3. Start talking with agent at your own."
            ]

            # Renderizar la lista de instrucciones
            for instruccion in instrucciones:
                st.markdown(instruccion)

      

        st.subheader(":file_folder: PDF File's Section")
        
        use_sample_pdf = st.checkbox("Talk with Imanol Asolo Bot")
        if use_sample_pdf:
            sample_pdf_path = os.path.join(os.getcwd(), "imanolpdf1.pdf")
            st.session_state.pdf_files = [sample_pdf_path]
        else:
            st.session_state.pdf_files = st.file_uploader("Talk with your own trained agents", type=['pdf'], accept_multiple_files=True)

        #st.session_state.api_key = st.text_input("Enter your OpenAI API key:")
        train = st.button("Train the Agent")
        if train:
            with st.spinner("Processing"):
                # get the text from PDFs
                raw_text = get_pdf_text(st.session_state.pdf_files)
                st.session_state.pdf_text = raw_text
                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                # create vector store
                vector_store = get_vector_store(text_chunks)
                # conversation chain
                st.session_state.conversation = get_conversation_chain(vector_store)
                # set train to True to indicate agent has been trained
                st.session_state.train = True
        st.subheader(":question: Questions that you can ask to the agent")
        with st.expander("Expand questions"):
            # Lista de instrucciones
            instrucciones = [
                "1. Who is Imanol Asolo?.",
                "2. Speak about Imanol Asolo´s skills.",
                "3. Give me a brief explanation about projects of Imanol Asolo.",
                "4. How can I contact with Imanol Asolo?"
            ]

            # Renderizar la lista de instrucciones
            for instruccion in instrucciones:
                st.markdown(instruccion)

    if not st.session_state.train:
        st.warning("First Train the Agent")

    if st.session_state.train:
        st.write("<h5><br>Ask anything from your documents, doesn´t matter the language I am multi-idiomatic !:</h5>", unsafe_allow_html=True)
        user_question = st.text_input(label="", placeholder="Enter something...")
        if user_question:
            handle_userInput(user_question)

if __name__ == "__main__":
    main()
