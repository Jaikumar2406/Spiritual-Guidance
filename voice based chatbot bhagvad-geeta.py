from dotenv import load_dotenv
import os

load_dotenv()
pine_cone = os.getenv('pine_cone')
groq = os.getenv('groq')
hugging_face = os.getenv('hugging_face')

from langchain_community.embeddings import HuggingFaceBgeEmbeddings

embedding = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"token": hugging_face})

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

pc = Pinecone(api_key=pine_cone)

index_name = 'bhagvad-geeta'
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-west-1")
    )

index = pc.Index(index_name, host=os.getenv('host_bg'))

from langchain.schema import Document

# docs = [Document(page_content=chunk) for chunk in chunks]

vstore = PineconeVectorStore(
    index=index,
    embedding=embedding,
    text_key='page_content'
)

from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain

model = ChatGroq(api_key=groq, temperature=0.5, model="llama-3.3-70b-versatile")

retriver_prompt = (
    "Given a chat history and the latest user question which might reference context in the chat history,"
    "formulate a standalone question which can be understood without the chat history."
    "Do NOT answer the question, just reformulate it if needed and otherwise return it as is.")

retriver = vstore.as_retriever(search_kwargs={"k": 3})

from langchain_core.prompts import ChatPromptTemplate

contrextual_qa = ChatPromptTemplate.from_messages(
    [
        ("system", retriver_prompt),
        MessagesPlaceholder(variable_name='chat_history'),
        ("human", '{input}')
    ]
)

histroy_aware_retriver = create_history_aware_retriever(model, retriver, contrextual_qa)

BOT = """You are a spiritual guide trained only on the teachings of the Bhagavad Gita. Whenever a user shares a personal problem, confusion, or question about life, relationships, stress, studies, or decisions — you respond with deep understanding, kindness, and clarity using the teachings of the Bhagavad Gita.

Your answer must:

Explain what Lord Krishna has said in the Gita about such situations.

Mention the chapter and verse (shloka number) related to it.

Share a brief, simple explanation of that shloka in easy words.

Give a small real-life example or story if possible.

End with emotional support, like a gentle friend or guide.

Use simple English. Do not use hard words. Be emotionally warm and connected to the user.

If the user asks anything not related to life problems or not connected with the Gita, gently tell them:

"Dear friend, I am trained to speak only about life and how the Bhagavad Gita can help us in difficult times. I cannot answer this question. But I am here to support you whenever you need guidance."

Your tone must be peaceful, humble, loving, and inspired by Lord Krishna's teachings and give answer between 100 to 500 words and also suggest what i do next like what are the my next step.

CONTEXT:
{context}

QUESTION: {input}

YOUR ANSWER:
"""

qa_bot = ChatPromptTemplate.from_messages(
    [
        ("system", BOT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ]
)

question_answer_chain = create_stuff_documents_chain(model, qa_bot)

chain = create_retrieval_chain(histroy_aware_retriver, question_answer_chain)

chat_history = []
store = {}

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


chain_with_memmory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# Importing voice-related libraries
import speech_recognition as sr
import pyttsx3

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Set properties for the voice (optional)
engine.setProperty('rate', 150)  # Speed of speech
engine.setProperty('volume', 0.9)  # Volume level (0.0 to 1.0)


# Function to get voice input
def get_voice_input():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening... (say 'stop' to end voice input)")
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            print(f"You (voice): {text}")
            return text
        except sr.UnknownValueError:
            print("Sorry, I didn't catch that. Please try again.")
            return None
        except sr.RequestError:
            print("Sorry, speech service is down. Please try typing instead.")
            return None


# Function to speak the response
def speak_response(text):
    print(f"AI: {text}")
    engine.say(text)
    engine.runAndWait()


# Main interaction loop
while True:
    print("\nChoose input method:")
    print("1. Type your question")
    print("2. Speak your question")
    print("3. Exit")

    choice = input("Enter your choice (1/2/3): ")

    if choice == '3':
        print("Hare Krishna. May Lord Krishna's blessings be with you.")
        break

    if choice not in ['1', '2']:
        print("Invalid choice. Please try again.")
        continue

    if choice == '1':
        user_input = input("You (text): ")
    elif choice == '2':
        user_input = get_voice_input()
        if user_input is None:
            continue

    if user_input.lower() in ["exit", "quit", "stop"]:
        speak_response("Hare Krishna. May Lord Krishna's blessings be with you.")
        break

    try:
        response = chain_with_memmory.invoke(
            {"input": user_input},
            config={
                "configurable": {"session_id": "143"}
            },
        )

        # Speak the response first, then print it
        speak_response(response["answer"])

    except Exception as e:
        error_msg = f"Sorry, an error occurred: {str(e)}"
        print(error_msg)
        engine.say(error_msg)
        engine.runAndWait()
        speak_response("I encountered an error. Please try again.")