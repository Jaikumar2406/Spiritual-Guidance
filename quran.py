from dotenv import load_dotenv
import os
load_dotenv()

with open("The-Quran-Saheeh-International.txt", "r", encoding="utf-8") as file:
    data = file.read()

chunks = [data[i:i+850] for i in range(0 , len(data),850)]

pine_cone = os.getenv('pine_cone')
groq = os.getenv('groq')
hugging_face = os.getenv('hugging_face')

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
embedding = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"token": hugging_face} )


from pinecone import Pinecone , ServerlessSpec
from langchain_pinecone import PineconeVectorStore

pc = Pinecone(api_key = pine_cone)

index_name = 'quarn'
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-west-1")
    )


index = pc.Index(index_name , host=os.getenv('host_qr'))


from langchain.schema import Document
docs = [Document(page_content=chunk) for chunk in chunks]


vstore = PineconeVectorStore(
    index = index,
    embedding=embedding,
    text_key='page_content'
)

# from tqdm import tqdm
# batch_size = 32  # you can tune this
# for i in tqdm(range(0, len(docs), batch_size)):
#     batch = docs[i:i+batch_size]
#     try:
#         vstore.add_documents(batch)
#     except Exception as e:
#         print(f"Error in batch {i}-{i+batch_size}: {e}")



from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import MessagesPlaceholder , ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain



model = ChatGroq(api_key = groq , temperature=0.5 , model ="llama-3.3-70b-versatile")


retriever_prompt = ("Given a chat history and the latest user question which might reference context in the chat history,"
    "formulate a standalone question which can be understood without the chat history."
    "Do NOT answer the question, just reformulate it if needed and otherwise return it as is.")


retriver = vstore.as_retriever(search_kwargs = {"k":3})

contextualize_bot = ChatPromptTemplate.from_messages(
    [
        ('system',retriever_prompt),
        MessagesPlaceholder(variable_name = "chat_history"),
        ("human",'{input}')
    ]
)
contextualize = create_history_aware_retriever(model , retriver , contextualize_bot)


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


qa_bot = ChatPromptTemplate.from_messages(
    [
        ("system",BOT),
        MessagesPlaceholder(variable_name = "chat_history"),
        ("human",'{input}')
    ]
)


stuff = create_stuff_documents_chain(model , qa_bot)

chain = create_retrieval_chain(contextualize , stuff)

chat_history = []
store = {}

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory


def get_session_history(session_id: str)-> BaseChatMessageHistory:
  if session_id not in store:
    store[session_id]= ChatMessageHistory()
  return store[session_id]

chain_with_memmory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)


while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit", "stop"]:
        break

    response = chain_with_memmory.invoke(
        {"input": user_input},
        config={
            "configurable": {"session_id": "143"}
        },
    )

    print("AI:", response["answer"])
