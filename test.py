import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain.schema.messages import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

# API Key untuk Google Generative AI
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
file_path = "pdf/pertamina.pdf"

# 1. Muat PDF
loader = PyPDFLoader(file_path)
documents = loader.load()

# 2. Split Dokumen
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_documents = text_splitter.split_documents(documents)

# 3. Buat Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)

# 4. Bangun Vectorstore
vectorstore = FAISS.from_documents(split_documents, embeddings)
retriever = vectorstore.as_retriever()

# 5. Siapkan Model LLM dan Prompt
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)

messages = [
     SystemMessage(
         content="""Kamu adalah Chatbot assistant yang bertugas untuk memberikan jawaban terkait sustainability report."""
     ),
     HumanMessage(content="Bagaimana sustainability report PT Pertamina?"),
 ]
llm.invoke(messages)

# system_prompt = (
#     "Anda adalah asisten ahli dalam laporan keberlanjutan. Gunakan konteks yang diberikan untuk menjawab pertanyaan. "
#     "Jika konteks tidak memiliki informasi yang cukup, jawab dengan 'Saya tidak tahu'."
#     "\n\n"
#     "{context}"
# )
# prompt = ChatPromptTemplate.from_messages([
#     ("system", system_prompt),
#     ("human", "{input}")
# ])

# # 6. Buat RAG Chain
# question_answer_chain = create_stuff_documents_chain(llm, prompt, document_variable_name="context")
# rag_chain = create_retrieval_chain(retriever, question_answer_chain)



question = "Apa itu Pertamina?"



