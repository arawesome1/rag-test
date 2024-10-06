import os 
from dotenv import load_dotenv
load_dotenv()

os.environ['langchain_api_key'] = os.getenv("langchain_api_key")
os.environ['langchain_tracing_V2'] = 'True'
os.environ['hf_token'] = os.getenv("hf_token")

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from langchain_faiss import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_huggingace import HuggingfaceEmbeddings
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_document import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever

# Initialize Huggingface embeddings
embeddings = HuggingfaceEmbeddings(model_name="all-MiniLM-L6-V2")

# Streamlit: Request the API key
api_key = st.text_input("Enter Your Groq Api_Key: ", type='password')

# System prompt for reformulating questions
system_prompt = (
    "Given a chat history and the latest user question which might reference context in the chat history, "
    "reformulate a standalone question that can be understood with the chat history. "
    "Do not answer the question, just reformulate it if needed; otherwise, return it as is."
)

# Q&A prompt template
qna_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ('user', "question:{question}")
])

# Initialize Chroma vectorstore
vectorstore = FAISS.from_documents([], embeddings)
retriever = vectorstore.as_retriever()

# Global RAG chain variable
rag_chain = None

# Initialize RAG chain with the selected engine
def initialize_rag_chain(engine):
    global rag_chain
    llm = ChatGroq(groq_api_key=api_key, model=engine)
    history_aware_retriever = create_history_aware_retriever(llm, retriever, qna_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qna_prompt)
    
    # Set the global rag_chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Generate response from the RAG chain
def generate_response(question, engine, temperature, max_token, session_id):
    global rag_chain
    # Initialize the chain if not already done
    if rag_chain is None:
        initialize_rag_chain(engine)

    chat_history = get_session_history(session_id)
    answer = rag_chain.invoke({'question': question, 'chat_history': chat_history.messages})

    # Add user and assistant messages to the chat history
    chat_history.add_message("user", question)
    chat_history.add_message("assistant", answer)

    return answer

# In-memory storage for chat histories
store = {}

# Retrieve session history
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Initialize conversational RAG chain with message history
def setup_conversational_rag_chain():
    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_message_key="input",
        history_message_key="chat_history",
        output_message_key="answer"
    )

# Streamlit UI
st.title("QnA Chatbot")

if api_key:
    # Sidebar settings for model selection
    engine = st.sidebar.selectbox("Select Model", [
        'gemma2-9b-it', 'lama3-groq-70b-8192-tool-use-preview',
        'mixtral-8x7b-32768', 'llava-v1.5-7b-4096-preview'
    ])
    temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
    max_token = st.sidebar.slider("Max Token", min_value=50, max_value=300, value=150)

    st.sidebar.write("Made by Ankit")

    # Main question input
    st.write("Go ahead and ask your question")
    user_input = st.text_input("You: ")
    session_id = "session"

    if user_input:
        response = generate_response(user_input, engine, temperature, max_token, session_id)
        st.write(response)
    else:
        st.write("Please provide user input")
else:
    st.warning("Please enter Groq API key")

