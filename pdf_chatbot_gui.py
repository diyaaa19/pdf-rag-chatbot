import tkinter as tk
from tkinterdnd2 import TkinterDnD, DND_FILES

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

root.mainloop()
