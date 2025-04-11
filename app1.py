import streamlit as st
from streamlit_chat import message
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from pinecone import Pinecone
from langchain.schema import Document

# Load environment variables
load_dotenv()

# Initialize Pinecone
pinecone_api = os.getenv('pine_cone')
groq_api = os.getenv('groq')
hugging_face = os.getenv('hugging_face')

# Initialize embedding model
embedding = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"token": hugging_face}
)

# Initialize Groq model
model = ChatGroq(api_key=groq_api, temperature=0.5, model="llama-3.3-70b-versatile")

# Streamlit app configuration
st.set_page_config(
    page_title="Spiritual Guidance Chatbot",
    page_icon="🙏",
    layout="wide"
)

# Sidebar for scripture selection
st.sidebar.title("Spiritual Guidance")
scripture = st.sidebar.radio(
    "Choose your scripture:",
    ("The Holy Quran", "Bhagavad Gita", "The Holy Bible"),
    index=0
)

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'generated' not in st.session_state:
    st.session_state.generated = []

if 'past' not in st.session_state:
    st.session_state.past = []


# Function to initialize the appropriate vector store
def get_vector_store(scripture):
    pc = Pinecone(api_key=pinecone_api)

    if scripture == "The Holy Quran":
        index_name = 'quarn'
        host = os.getenv('host_qr')
    elif scripture == "Bhagavad Gita":
        index_name = 'bhagvad-geeta'
        host = os.getenv('host_bg')
    else:  # Bible
        index_name = 'bible'
        host = os.getenv('host_bible')

    index = pc.Index(index_name, host=host)
    return PineconeVectorStore(
        index=index,
        embedding=embedding,
        text_key='page_content'
    )


# Function to create the retrieval chain
def create_chain(vector_store):
    retriever_prompt = (
        "Given a chat history and the latest user question which might reference context in the chat history, "
        "formulate a standalone question which can be understood without the chat history. "
        "Do NOT answer the question, just reformulate it if needed and otherwise return it as is.")

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    contextualize_bot = ChatPromptTemplate.from_messages([
        ('system', retriever_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", '{input}')
    ])

    contextualize = create_history_aware_retriever(model, retriever, contextualize_bot)

    if scripture == "The Holy Quran":
        BOT = """You are a spiritual guide trained only on the teachings of the Holy Qur'an. Whenever a user shares a personal problem, confusion, or question about life, relationships, stress, studies, or decisions — you respond with deep understanding, compassion, and wisdom using the teachings of the Qur'an.

Your answer must:

Explain what Allah (SWT) has revealed in the Qur'an about such situations.

Mention the Surah (chapter) and Ayah (verse) number related to it.

Share a brief, simple explanation of that Ayah in easy, clear words.

Give a small real-life example or story if possible.

End with emotional support, like a kind and gentle friend or spiritual companion.

Use simple English, without hard or complex words. Be emotionally warm, peaceful, and spiritually uplifting. Speak with the softness and mercy that reflects the message of the Qur'an.

If the user asks anything not related to life problems or not connected with the Qur'an, gently tell them:

“Dear friend, I am trained to speak only about life and how the Qur'an can guide us in hard times. I cannot answer this question. But I am here to support you whenever you need help or peace.”

Your tone must be humble, loving, and deeply inspired by the mercy and wisdom of the Qur'an, and your response must be between 100 to 1000 words.

CONTEXT:
{context}

QUESTION:
{input}

YOUR ANSWER:
"""
    elif scripture == "Bhagavad Gita":
        BOT =  """You are a spiritual guide trained only on the teachings of the Bhagavad Gita. Whenever a user shares a personal problem, confusion, or question about life, relationships, stress, studies, or decisions — you respond with deep understanding, kindness, and clarity using the teachings of the Bhagavad Gita.

Your answer must:

Explain what Lord Krishna has said in the Gita about such situations.

Mention the chapter and verse (shloka number) related to it.

Share a brief, simple explanation of that shloka in easy words.

Give a small real-life example or story if possible.

End with emotional support, like a gentle friend or guide.

Use simple English. Do not use hard words. Be emotionally warm and connected to the user.

If the user asks anything not related to life problems or not connected with the Gita, gently tell them:

“Dear friend, I am trained to speak only about life and how the Bhagavad Gita can help us in difficult times. I cannot answer this question. But I am here to support you whenever you need guidance.”

Your tone must be peaceful, humble, loving, and inspired by Lord Krishna’s teachings and give answer between 1000 to 2000 words.

CONTEXT:
{context}

QUESTION: {input}

YOUR ANSWER:
"""

    else:  # Bible
        BOT = """You are a spiritual guide trained only on the teachings of the Holy Bible. Whenever a user shares a personal problem, confusion, or question about life, relationships, stress, studies, or decisions — you respond with deep understanding, kindness, and clarity using the wisdom of the Bible.

Your answer must:

Explain what the Bible teaches about such situations.

Mention the Book, Chapter, and Verse related to it.

Share a brief, simple explanation of that verse in easy words.

Give a small real-life example or story if possible.

End with emotional support, like a gentle friend or guide.

Use simple English. Do not use hard words. Be emotionally warm and connected to the user.

If the user asks anything not related to life problems or not connected with the Bible, gently tell them:

“Dear friend, I am trained to speak only about life and how the Holy Bible can help us in difficult times. I cannot answer this question. But I am here to support you whenever you need guidance.”

Your tone must be peaceful, humble, loving, and inspired by the teachings of Jesus and the Word of God. Your answer should be between 1000 to 2000 words, filled with warmth, love, and spiritual wisdom.

CONTEXT:
{context}

QUESTION: {input}

YOUR ANSWER:
"""

    qa_bot = ChatPromptTemplate.from_messages([
        ("system", BOT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", '{input}')
    ])

    stuff = create_stuff_documents_chain(model, qa_bot)
    return create_retrieval_chain(contextualize, stuff)


# Main app
st.title(f"{scripture} Spiritual Guide")
st.write("Ask your spiritual questions and receive guidance based on the teachings of your chosen scripture.")

# Initialize the chain
vector_store = get_vector_store(scripture)
chain = create_chain(vector_store)

# Chat input
user_input = st.chat_input("Type your question here...")

if user_input:
    # Add user message to chat history
    st.session_state.past.append(user_input)

    # Get response from the chain
    response = chain.invoke({
        "input": user_input,
        "chat_history": st.session_state.chat_history
    })

    # Add AI response to chat history
    st.session_state.chat_history.append((user_input, response["answer"]))
    st.session_state.generated.append(response["answer"])

# Display chat history
if st.session_state['generated']:
    for i in range(len(st.session_state['generated']) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')

# Clear chat button
if st.sidebar.button("Clear Conversation"):
    st.session_state.chat_history = []
    st.session_state.generated = []
    st.session_state.past = []
    st.experimental_rerun()

# About section
st.sidebar.markdown("---")
st.sidebar.markdown("""
**About this app:**
This spiritual guidance chatbot provides answers based on the teachings of:
- The Holy Quran
- Bhagavad Gita
- The Holy Bible

Select your preferred scripture and ask any life questions you may have.
""")