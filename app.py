
import os
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import torch
import faiss

from huggingface_hub import login
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Smart Article Insights Generator",
    page_icon="🧠",
    layout="wide"
)


# --------------------------------------------------
# 2. HUGGING FACE LOGIN
# --------------------------------------------------

hf_token = os.environ.get("HF_TOKEN")

if hf_token:
    try:
        login(
            token=hf_token,
            add_to_git_credential=False
        )
    except Exception as e:
        st.warning(f"Hugging Face login failed: {e}")


# --------------------------------------------------
# 3. LOAD TINYLLAMA MODEL
# --------------------------------------------------

@st.cache_resource
def load_tinyllama_model():

    # Import inside the function so the app can start cleanly and
    # show the real model-loading error if Transformers has a problem.
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True
    )

    model.eval()

    return tokenizer, model


# --------------------------------------------------
# 4. LOAD ARTICLES DATA
# --------------------------------------------------

@st.cache_data
def load_articles_data():

    file_path = "Articles.csv"

    try:
        data = pd.read_csv(file_path)
        return data

    except FileNotFoundError:
        st.error(
            "Articles.csv not found. "
            "Please make sure Articles.csv is present in the project folder."
        )

        return pd.DataFrame({
            "Title": [],
            "Article": []
        })


articles_df = load_articles_data()


# --------------------------------------------------
# 5. SELECT SAMPLE ARTICLE FOR RAG
# --------------------------------------------------

if not articles_df.empty and len(articles_df) > 5:

    rag_sample_article = articles_df["Article"].iloc[5]

else:

    rag_sample_article = (
        "No article available for RAG demonstration."
    )


# --------------------------------------------------
# 6. SETUP RAG COMPONENTS
# --------------------------------------------------

@st.cache_resource
def setup_rag_components(article_text):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = text_splitter.split_text(article_text)

    embedder = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    chunk_embeddings = embedder.encode(chunks)

    d = chunk_embeddings.shape[1]

    index = faiss.IndexFlatL2(d)

    index.add(
        np.array(chunk_embeddings).astype("float32")
    )

    return embedder, index, chunks


with st.spinner("Preparing RAG components..."):

    embedder, faiss_index, article_chunks = setup_rag_components(
        rag_sample_article
    )


# --------------------------------------------------
# 7. LAZY LOAD TINYLLAMA
# --------------------------------------------------

tokenizer_tinyllama = None
model_tinyllama = None


def get_tinyllama():
    global tokenizer_tinyllama, model_tinyllama

    if tokenizer_tinyllama is None or model_tinyllama is None:
        with st.spinner("Loading TinyLlama model..."):
            tokenizer_tinyllama, model_tinyllama = load_tinyllama_model()

    return tokenizer_tinyllama, model_tinyllama


# --------------------------------------------------
# 7. SUMMARIZATION FUNCTION
# --------------------------------------------------

def summarize_tinyllama(article):

    tokenizer_tinyllama, model_tinyllama = get_tinyllama()

    messages = [
        {
            "role": "system",
            "content": "You are an AI assistant that summarizes articles clearly and concisely."
        },
        {
            "role": "user",
            "content": f"Summarize this article in 3 to 5 sentences:\n\n{article}"
        }
    ]

    inputs = tokenizer_tinyllama.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )

    outputs = model_tinyllama.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer_tinyllama.eos_token_id
    )

    input_length = inputs["input_ids"].shape[-1]

    generated_tokens = outputs[0][input_length:]

    summary = tokenizer_tinyllama.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    return summary


# --------------------------------------------------
# 8. RAG RETRIEVAL
# --------------------------------------------------

def retrieve_chunks(query, k=3):

    query_embedding = embedder.encode([query])

    k = min(k, len(article_chunks))

    distances, indices = faiss_index.search(
        np.array(query_embedding).astype("float32"),
        k
    )

    retrieved_chunks = [
        article_chunks[i]
        for i in indices[0]
        if i >= 0
    ]

    return retrieved_chunks


# --------------------------------------------------
# 9. RAG QUESTION ANSWERING
# --------------------------------------------------

def answer_question_with_rag(article_text, question, k=3):

    tokenizer_tinyllama, model_tinyllama = get_tinyllama()

    retrieved_context = retrieve_chunks(question, k=k)

    context_text = "\n".join(retrieved_context)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful question-answering assistant. "
                "Answer only using the provided context. "
                "If the answer is not present in the context, say "
                "\"I don't know based on the provided context.\""
            )
        },
        {
            "role": "user",
            "content": (
                f"Context:\n{context_text}\n\n"
                f"Question: {question}\n\n"
                "Answer clearly and concisely."
            )
        }
    ]

    inputs = tokenizer_tinyllama.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )

    outputs = model_tinyllama.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer_tinyllama.eos_token_id
    )

    input_length = inputs["input_ids"].shape[-1]

    generated_tokens = outputs[0][input_length:]

    answer = tokenizer_tinyllama.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    return answer