from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document

# Load Bible text
with open("pdfs/bible.txt") as f:
    text = f.read()

# Split
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = [Document(page_content=chunk) for chunk in splitter.split_text(text)]

# Embed & save
Chroma.from_documents(docs, OpenAIEmbeddings(), persist_directory="chroma_db")