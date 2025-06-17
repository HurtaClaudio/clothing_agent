from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI


vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

llm = ChatOpenAI(model="gpt-4o", temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # or "map_reduce", "refine" depending on use case
    retriever=retriever,
    return_source_documents=True
)