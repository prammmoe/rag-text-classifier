import getpass
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

from langchain.chains import RetrievalQA  # Import RetrievalQA

from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

file_path = "pdf/pertamina.pdf"
loader = PyPDFLoader(file_path)

documents = []

docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents.extend(text_splitter.split_documents(docs))
# splits = text_splitter.split_documents(docs)

print(f"Total loaded document chunks: {len(documents)}")

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)

print(llm)

valid_documents = []
for doc in documents:
    if doc.page_content.strip():  # Pastikan konten tidak kosong
        try:
            embedding = embeddings.embed_query(doc.page_content)
            if embedding:  # Hanya tambahkan jika embedding valid
                valid_documents.append(doc)
        except Exception as e:
            print(f"Error generating embedding for doc: {e}")

if not valid_documents:
    raise ValueError("No valid documents to embed!")

print(valid_documents)

vectorstore = FAISS.from_documents(valid_documents, embeddings)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

def generate_rag_prompt(query, context):
    prompt = f"""
Anda adalah bot yang berperan sebagai ahli sustainability yang dapat memberikan jawaban informatif berdasarkan konteks teks yang tersedia di bawah ini.
Jawablah pertanyaan dengan kalimat lengkap yang menyeluruh, mencakup semua informasi latar belakang yang relevan.
Ingat, Anda berbicara kepada audiens non-teknis, jadi jelaskan konsep yang rumit dengan cara yang sederhana dan gunakan nada yang ramah.
PERTANYAAN: '{query}'
KONTEKS: '{context}'
JAWABAN:
"""
    return prompt

# Buat template dalam bahasa Indonesia untuk chain RetrievalQA
template = """
Anda adalah seorang ahli sustainability yang berpengalaman dalam menjelaskan jawaban akurat dari teks yang kompleks.
Manfaatkan konteks yang diberikan untuk memberikan jawaban yang jelas dan terinci.

Konteks:
{context}

Berikan jawaban yang informatif dan mendalam berdasarkan konteks yang ada:
"""

prompt = ChatPromptTemplate.from_messages([("system", template), ("human", "{input}")])

chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs={"prompt": prompt})

while True:
    query = input("Query (or type 'exit' to quit): ")
    if query.lower() == 'exit':  # Check if the user wants to exit
        print("Exiting the program. Goodbye!")
        break  # Exit the loop

    context = "\n".join([result.page_content for result in retriever.get_relevant_documents(query)])
    prompt = generate_rag_prompt(query=query, context=context)

    # Create a HumanMessage object with the generated prompt
    messages = [HumanMessage(content=prompt)]

    # Pass the messages to the llm
    answer = llm(messages=messages)
    print("Answer:", answer.content)