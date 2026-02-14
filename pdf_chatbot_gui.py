import tkinter as tk
from tkinterdnd2 import TkinterDnD, DND_FILES
from langchain_community.chat_models import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from tkinter import scrolledtext


vector_db = None
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2
)

root = TkinterDnD.Tk()
root.title("RAG PDF Chatbot")
root.geometry("700x600")

title = tk.Label(root, text="RAG PDF Chatbot", font=("Arial", 16))
title.pack(pady=10)

pdf_entry = tk.Entry(root, width=70)
pdf_entry.pack(pady=10)

def drop(event):
    file_path = event.data.strip("{}")
    pdf_entry.delete(0, tk.END)
    pdf_entry.insert(0, file_path)

pdf_entry.drop_target_register(DND_FILES)
pdf_entry.dnd_bind("<<Drop>>", drop)

def load_pdf_text(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)
    return chunks

def create_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings

def create_vectorstore(chunks, embeddings):
    db = FAISS.from_documents(chunks, embeddings)
    return db

def load_pdf():
    global vector_db

    file_path = pdf_entry.get()

    documents = load_pdf_text(file_path)
    chunks = chunk_documents(documents)
    embeddings = create_embeddings()
    vector_db = create_vectorstore(chunks, embeddings)

load_button = tk.Button(root, text="Load PDF", command=load_pdf)
load_button.pack(pady=5)
chat_box = scrolledtext.ScrolledText(root, width=80, height=20)
chat_box.pack(pady=10)
question_entry = tk.Entry(root, width=60)
question_entry.pack(side=tk.LEFT, padx=5, pady=5)
ask_button = tk.Button(root, text="Ask")
ask_button.pack(side=tk.LEFT)


root.mainloop()
