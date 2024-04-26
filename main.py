import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI 
from htmlTemplates import css, bot_template, user_template
import os
import subprocess

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
    st.set_page_config(page_title="Imanol Asolo portfolio", page_icon=":clipboard:")
    col1, col2 = st.columns([1,3])
    with col1:
        st.image('foto_imanol.jpg', width=100)
    with col2: 
        st.title("Welcome to my portfolio!")

    # Menú de navegación
    menu = {
        "Home": render_home,
        "About Imanol Asolo": render_about,
        "Skills": render_skills,
        "Projects": render_projects,
        "Contact": render_contact,
        "Chat with Imanol Asolo´s bot": render_chat,
    }

    #st.sidebar.markdown('<a href="https://imanol-asolo-ai-chat.streamlit.app/" target="_blank">Chat with Imanol Asolo´s AI</a>', unsafe_allow_html=True)

    # Renderizar la lista de enlaces como botones
    for opcion, render_func in menu.items():
        if opcion == "Chat":
            st.sidebar.button(opcion, on_click=render_func)
        else:
            if st.sidebar.button(opcion):
                render_func()


def render_home():
    st.warning("Imanol Asolo is not your average Full Stack Developer. With a passion for crafting exceptional digital experiences and a knack for driving agile project management, Imanol stands at the intersection of technical prowess and effective leadership. As a certified Scrum Master, he brings a wealth of experience in guiding teams to success through collaboration, innovation, and a relentless pursuit of excellence. Dive into Imanol's portfolio to explore a world where code meets creativity, and where every project is an opportunity to make a meaningful impact.")

def render_about():
    st.success("Meet Imanol Asolo, a passionate Full Stack Developer and dedicated Scrum Master, shaping the digital landscape with innovation and expertise. Beyond the world of coding, Imanol is a devoted husband and proud father, finding balance and inspiration in family life. With a deep love for beach sports and the sea, he brings the same energy and enthusiasm to his work, creating seamless digital experiences that leave a lasting impression. Explore Imanol's portfolio to discover the perfect blend of technical excellence, leadership, and a touch of seaside charm.")

def render_skills():
    st.info("Immerse yourself in the world of technology with Imanol Asolo, a seasoned Full Stack Developer and visionary Scrum Master. With a wealth of experience in building robust web applications and leading agile development teams, Imanol combines technical prowess with strategic leadership to drive projects forward. From crafting elegant frontend interfaces to architecting scalable backend systems, he possesses a diverse skill set that fuels innovation and fosters collaboration. Dive into Imanol's portfolio to witness firsthand the fusion of technical excellence, agile methodologies, and a passion for pushing boundaries in the digital realm.")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image('python_icon.png',width=100)
        st.warning(':star::star::star::star: phyton')
    with col2:
        st.image('javascript_icon.png',width=100)
        st.warning(':star::star::star::star: JavaScript')
    with col3:
        st.image('unix_logo.png',width=100)
        st.warning(':star::star::star::star: Bash Scripting')

    col1, col2, col3 =st.columns(3)
    with col1:
        st.image('react_icon.png',width=100)
        st.warning(':star::star::star::star: React')
    with col2:
        st.image('django_icon.png',width=100)
        st.warning(':star::star::star::star: Django')   
    with col3:
        st.image('vuejs_icon.png',width=100)
        st.warning(':star::star::star: Vue JS')

    st.title("And more to come...")

    
def render_projects():
    st.warning("Embark on a journey of digital transformation with Imanol Asolo, a versatile Full Stack Developer and seasoned Scrum Master. Delve into an array of captivating projects that showcase Imanol's expertise in crafting cutting-edge solutions and driving agile development initiatives to success. From dynamic web applications to sophisticated software implementations, each project reflects Imanol's commitment to excellence, creativity, and strategic problem-solving. Explore the intersection of technology and innovation as you navigate through Imanol's project portfolio, where every endeavor represents a testament to his unwavering dedication to pushing the boundaries of possibility in the digital landscape.")
    col1, col2, col3 =st.columns(3)
    with col1:
        st.image("raptoreye_logo.png", width=100)
        if __name__ == "__main__":
            st.write("Click the button below to download the Raptor Eye presentation.")
        download_pdf_raptor_eye()
    with col2:
        st.image("AI_medicare_logo.png", width=100)
        if __name__ == "__main__":
            st.write("Click the button below to download the AI Medicare presentation.")
        download_pdf_ai_medicare()
    with col3:
        st.image("Chatbotter_logo.png", width=100)
        if __name__ == "__main__":
            st.write("Click the button below to download the Chatbotter challenge presentation.")
        #download_pdf_raptor_eye()

def download_pdf_raptor_eye():
    # Here you can put the code to generate or fetch the PDF file
    
        with open("Raptor_Eye_pres.pdf", "rb") as file:
            pdf_bytes = file.read()
        st.download_button(
            label="Download PDF",
            data=pdf_bytes,
            file_name="Raptor_Eye_pres.pdf",
            mime="application/pdf",
        )
    
def download_pdf_ai_medicare():
    # Here you can put the code to generate or fetch the PDF file
    
    with open("AI_medicare_pres.pdf", "rb") as file:
        pdf_bytes = file.read()
    st.download_button(
        label="Download PDF",
        data=pdf_bytes,
        file_name="AI_medicare_pres.pdf",
        mime="application/pdf",
        )
           

def render_contact():
    st.success("Ready to embark on a transformative journey fueled by innovation and expertise? Reach out to Imanol Asolo, a seasoned Full Stack Developer and adept Scrum Master, to explore synergies, spark conversations, and unlock new possibilities in the realm of technology and agile development. Whether you're seeking to kickstart a groundbreaking project, optimize your development processes, or simply exchange insights and ideas, Imanol welcomes the opportunity to connect, collaborate, and co-create value together. Drop a message, schedule a call, or send a carrier pigeon – whatever your preferred mode of communication, Imanol is here to listen, engage, and embark on a shared journey of growth and success. Let's connect and pave the way for innovation!")
    col1, col2, col3, col4 =st.columns(4)
    with col1:
        st.image('mail_icon.png', width=80)
        st.markdown('<a href="mailto:jjusturi@gmail.com">Send me a mail</a>', unsafe_allow_html=True)
           
    with col2:
        st.image('whatsapp_logo.png', width=100)
        st.markdown('<a href="https://wa.me/+5930993513082">Send a whatsapp message</a>', unsafe_allow_html=True)

    with col3:
        st.image('meeting_icon.png', width=100)
        st.markdown('<a href="https://buymeacoffee.com/imanolasolo">Let`s have a coffee and have a consultation about technical issues or Coach & Coffee!</a>', unsafe_allow_html=True)
    with col4:
        st.image('linkedin_logo.png', width=80)
        st.markdown('<a href="https://www.linkedin.com/in/imanolasolo/">Find me on Linkedin!</a>', unsafe_allow_html=True)

def render_chat():
    st.markdown('<a href="https://imanol-asolo-ai-chat.streamlit.app/">Chat with Imanol Asolo´s AI</a>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
