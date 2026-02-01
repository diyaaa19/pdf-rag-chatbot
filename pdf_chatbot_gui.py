import tkinter as tk
from tkinterdnd2 import TkinterDnD, DND_FILES
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

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
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )
    return embeddings

root.mainloop()
