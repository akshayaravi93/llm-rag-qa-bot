# from datasets import load_dataset # Using hugging face datasets 
# import time

# # for downloading the data from the internet
# max_retries = 3
# retry_delay = 5  # seconds

# for attempt in range(max_retries):
#     try:
#         dataset = load_dataset("cnn_dailymail", "3.0.0")
#         break  # Exit the loop if successful
#     except Exception as e:
#         print(f"Attempt {attempt + 1} failed: {e}")
#         if attempt < max_retries - 1:
#             print(f"Retrying in {retry_delay} seconds...")
#             time.sleep(retry_delay)
#         else:
#             raise  # Re-raise the exception if all retries fail
# dataset.save_to_disk("data/cnn_dailymail")

import numpy as np
import os

# To load dataset from disk
from datasets import load_from_disk

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA

from dotenv import load_dotenv

# Loading data from the disk
dataset = load_from_disk(dataset_path='data/cnn_dailymail')
print(dataset)
print(dataset['train']['article'][0])
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap=50,
)
texts = text_splitter.split_text(dataset["train"][0]["article"])
# print(texts)

###
embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",  # 384-dim, free
    model_kwargs={"device": "cpu"}  # No GPU needed
)
# print(embedder)

###
vector_db = FAISS.from_texts(texts, embedder)
vector_db.save_local("models/cnn_faiss_index")  # Reusable index

# Get embeddings from FAISS
embeddings = vector_db.index.reconstruct_n(0, vector_db.index.ntotal)

####

load_dotenv()  # Load HF_TOKEN from .env

llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token=os.getenv("HF_TOKEN")
)

###

qa_bot = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # Simple concatenation
    retriever=vector_db.as_retriever(search_kwargs={"k": 3})  # Top 3 chunks
)