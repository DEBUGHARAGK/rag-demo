import json
import os
import sys
import boto3
import streamlit as st

# Importing the required libraries for Bedrock and LangChain
from langchain_aws import BedrockEmbeddings
from langchain_community.llms import Bedrock

# Data Ingestion
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Vector Embedding and Vector Store
from langchain_community.vectorstores import FAISS

# LLM Models and Chains
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Initialize Bedrock Client
bedrock = boto3.client(service_name="bedrock-runtime")

# Using Titan Embeddings Model for generating embeddings
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

# Data ingestion function
def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()

    # Split the documents using Character Text Splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    return docs

# Function to create the vector store using FAISS
def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")

# Function to initialize the Titan Text G1 - Lite model
def get_titan_llm():
    # Create the model with the correct parameters for the Titan Text G1 - Lite LLM
    llm = Bedrock(
        model_id="amazon.titan-text-lite-v1",
        client=bedrock,
        model_kwargs={'maxTokenCount': 512}  # Ensure the correct parameter for the Titan model
    )
    return llm

# Define the prompt template
prompt_template = """
Human: Use the following pieces of context to provide a 
concise answer to the question at the end but use at least 250 words to summarize with detailed explanations. 
If you don't know the answer, just say that you don't know and don't try to make up an answer.
<context>
{context}
</context>

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# Function to get a response from the LLM
def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    # Directly pass only the query without extra arguments
    answer = qa({"query": query})
    return answer['result']

# Main function for the Streamlit application
def main():
    st.set_page_config(page_title="Chat PDF")
    
    st.header("Chat with PDF using AWS Bedrock 💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.title("Update or Create Vector Store:")
        
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Done")

    if st.button("Titan Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm = get_titan_llm()
            
            st.write(get_response_llm(llm, faiss_index, user_question))
            st.success("Done")

if __name__ == "__main__":
    main()