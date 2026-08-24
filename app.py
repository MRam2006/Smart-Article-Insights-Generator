
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

    prompt = (
        "Summarize the following article clearly and concisely:"
    )

    input_text = (
        f"{prompt}\n"
        f"{article}\n"
        f"Summary:"
    )

    inputs = tokenizer_tinyllama(
        input_text,
        return_tensors="pt",
        max_length=1024,
        truncation=True
    )

    outputs = model_tinyllama.generate(
    inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=150,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    pad_token_id=tokenizer_tinyllama.eos_token_id,
    )

    generated_text = tokenizer_tinyllama.decode(
        outputs[0],
        skip_special_tokens=True
    )

    summary_start_index = generated_text.find("Summary:")

    if summary_start_index != -1:

        summary = generated_text[
            summary_start_index + len("Summary:")
        ].strip()

    else:

        summary = generated_text.strip()

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

    retrieved_context = retrieve_chunks(
        question,
        k=k
    )

    context_text = "\n".join(
        retrieved_context
    )

    input_text = f"""
Based on the following context, answer the question.

If the answer is not in the context, state that you don't know.

Context:
{context_text}

Question:
{question}

Answer:
"""

    inputs = tokenizer_tinyllama(
        input_text,
        return_tensors="pt",
        max_length=1024,
        truncation=True,
        padding=True
    )

    outputs = model_tinyllama.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer_tinyllama.eos_token_id
    )

    generated_text = tokenizer_tinyllama.decode(
        outputs[0],
        skip_special_tokens=True
    )

    answer_start_index = generated_text.find("Answer:")

    if answer_start_index != -1:

        answer = generated_text[
            answer_start_index + len("Answer:")
        ].strip()

    else:

        answer = generated_text.strip()

    return answer


# --------------------------------------------------
# 10. STREAMLIT USER INTERFACE
# --------------------------------------------------

st.title("🧠 Smart Article Insights Generator")

st.markdown(
    """
    Summarize an article or ask a question about it.
    For Question Answering, RAG is applied to a pre-loaded sample article.
    """
)


mode = st.radio(
    "Select Mode",
    [
        "Summarize",
        "Answer Question (RAG)"
    ]
)


article_input = st.text_area(
    "Article Text",
    height=300,
    placeholder="Paste the article here..."
)


question_input = None


if mode == "Answer Question (RAG)":

    st.info(
        "For RAG-based Question Answering, "
        "the system uses a pre-loaded sample article."
    )

    st.markdown(
        "**RAG Sample Article (used for retrieval):**"
    )

    with st.expander("View RAG Sample Article"):

        st.write(rag_sample_article)

    question_input = st.text_input(
        "Question",
        placeholder="Enter your question here..."
    )


if st.button("🚀 Process"):

    if mode == "Summarize":

        if article_input:

            with st.spinner(
                "Generating summary..."
            ):

                output = summarize_tinyllama(
                    article_input
                )

            st.subheader("📝 Summary")
            st.write(output)

        else:

            st.warning(
                "Please provide an article to summarize."
            )


    elif mode == "Answer Question (RAG)":

        if question_input:

            with st.spinner(
                "Generating answer using RAG..."
            ):

                output = answer_question_with_rag(
                    rag_sample_article,
                    question_input
                )

            st.subheader("💡 Answer")
            st.write(output)

        else:

            st.warning(
                "Please provide a question to answer."
            )
