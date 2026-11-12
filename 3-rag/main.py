

import os
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question strictly "
    "based on the context provided below. If you do not know the answer, simply say that you don't know.\n\n"
    "Context:\n{context}"
)


def load_document(file_path: str) -> list[Document]:
    """Loads a PDF or TXT file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    docs = []

    if ext == ".pdf":
        reader = PdfReader(file_path)
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                docs.append(Document(
                    page_content=text,
                    metadata={"source": file_path, "page": i}
                ))
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        docs.append(Document(
            page_content=text,
            metadata={"source": file_path}
        ))
    else:
        raise ValueError(f"Unsupported file format: {ext}. Please use .pdf or .txt.")

    return docs


def rag_pipeline(
    input_file: str,
    llm_model: str = "qwen2.5",
    embed_model: str = "nomic-embed-text",
    temperature: float = 0.0,
    k: int = 5,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    db_dir: str = "./chroma_db",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
):
    """Builds and returns a ready-to-use RAG chain based on the provided parameters."""
    # 1. Embedding model
    embeddings = OllamaEmbeddings(model=embed_model)

    # 2. Vector store (load from disk or create a new one)
    if os.path.exists(db_dir) and os.listdir(db_dir):
        print(f"Loading existing vector database from '{db_dir}'...")
        vectorstore = Chroma(persist_directory=db_dir, embedding_function=embeddings)
    else:
        print(f"Processing file '{input_file}' and creating a new database...")
        docs = load_document(input_file)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        splits = text_splitter.split_documents(docs)

        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=db_dir
        )

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # 3. Main language model and prompt
    llm = ChatOllama(model=llm_model, temperature=temperature)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 4. Combine into a RAG chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain


if __name__ == "__main__":
    # Example call with custom parameters:
    rag_chain = rag_pipeline(
        input_file="sherlock.txt",
        llm_model="qwen",
        embed_model="nomic-embed-text",
        temperature=0.0,
        k=5,
        chunk_size=1000,
        chunk_overlap=200,
        db_dir="./chroma_sherlock"
    )

    while True:
        exit_keyword = "thanks"
        query = input(f"\nAsk a question (or type '{exit_keyword}' to exit): ")
        if query.strip().lower() == exit_keyword:
            break
        if not query.strip():
            continue

        response = rag_chain.invoke({"input": query})
        print()
        print("AI RESPONSE: {}", response["answer"])

