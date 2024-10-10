import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
from dotenv import load_dotenv
load_dotenv()
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ['hf_token']= os.getenv('hf_token')
embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
st.title("Conversational with PDF")
st.write("Upload your PDF and Chat with it")

api_key = st.text_input("Enter your Groq API Key", type="password")
if api_key:
    st.write("API Key entered")
    llm = ChatGroq(groq_api_key=api_key, model_name='gemma2-9b-it')
    session_id = st.text_input("Session ID", value="default_session")
    if 'store' not in st.session_state:
        st.session_state.store = {}
    uploaded_files=st.file_uploader('Choose a PDF File', type='pdf',accept_multiple_files=True)
    if uploaded_files:
        document=[]
        for uploaded_file in uploaded_files:
            temppdf=f"./temp.pdf"
            with open(temppdf, "wb") as file:
                file.write(uploaded_file.getvalue())
                file_name=uploaded_file.name
                loader= PyPDFLoader(temppdf)
                docs=loader.load()
                document.extend(docs)
                text_splitter=RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=200)
        splits=text_splitter.split_documents(document)
        vectorstore=Chroma.from_documents(documents = docs,embedding=embeddings)
        retriever = vectorstore.as_retriever()
        contextualize_q_prompt=("""Given a chat history and latest user question which 
                                might reference context in chat history, formulate a standalone question 
                                which can be understood without the chat history. DO NOT answer the question
                                just reformulate it if needed and otherwise return it as is """)
        contextualize_prompt=ChatPromptTemplate.from_messages([
            ('system',contextualize_q_prompt),
            MessagesPlaceholder("chat_history"),
            ('human','{input}'),
        ])
        history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_prompt)

        system_prompt=("""You are an assistance created by Ankit and is for helpin for question answerin task
                       use the following pieces of retriever content to answer the question if you dont know the answer
                       say that you don't know.keep the answer concise
                       \n\n
                       {context}""")
        qa_prompt= ChatPromptTemplate.from_messages([
            ('system',system_prompt),
            MessagesPlaceholder("chat_history"),
            ('human','{input}'),
        ])
        question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=qa_prompt)
        rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)

        def get_Session_history(session:str)-> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]
        
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_Session_history,
            input_messages_key='input',
            history_messages_key='chat_history',
            output_messages_key='answer'
        )
        user_input = st.text_input("Your Question: ")
        if user_input:
            session_history= get_Session_history(session_id)
            response= conversational_rag_chain.invoke({'input':user_input},
                                                      config={'configurable':{'session_id': session_id}
                                                              })
            st.write("Assistant: ", response['answer'])
else:
    st.warning("Please enter Groq API Key")
    description = """
    Made by Ankit \n\n
    The **Conversational PDF Assistant** allows users to upload PDF documents and engage in interactive conversations with an AI model. 
    Users can ask questions about the content of the PDFs, and the application responds with accurate, context-aware answers, 
    enhancing information retrieval and making document exploration intuitive and user-friendly.

    To access the Groq API for processing and generating responses, you will need a Groq API key. This key ensures secure and authenticated communication with the Groq services. Privacy is maintained as user data is handled securely and not stored beyond the session.

    You can generate your Groq API key by visiting [this link](https://console.groq.com/keys).
    """

    st.markdown(description)
