from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from utils import read_pdf
import os
from dotenv import load_dotenv
load_dotenv()


def create_index():
    # Load the documents
    directory = "my_data_directory"
    loader = PyPDFDirectoryLoader(directory)
    documents = loader.load()
    print("Number of original documents: ", len(documents))  # It says 3 because one page is one document (metadata contains page_number)
    print("Document example: ", documents[-1])

    # Document transformer
    chunk_size = 1_000
    chunk_overlap = 20
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    print("Documents length: ", len(docs))

    # Create data embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # We will use only first two docs initially, and then the last two, but this is just to show how can Pinecone perform lazy adding
    index_name = "my_index"
    index_pc = PineconeVectorStore.from_documents(docs, embedding=embedding_model, index_name=index_name)

    print("Documents have been saved!")


def test_index():
    def get_similar_docs(index_pc, query, k=2):
        similar_docs = index_pc.similarity_search(query, k=k)
        return similar_docs

    index_name = "my_index"
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    index_pc = PineconeVectorStore.from_existing_index(embedding=embedding_model, index_name=index_name)
    user_document = read_pdf("my_file")
    query_index = f"Retrieve the tips on how to write a CV! They should be related to: {user_document}"

    context_docs = get_similar_docs(index_pc, query_index, k=5)

    print("Docs: ", context_docs)


if __name__ == "__main__":
    # create_index()
    test_index()
