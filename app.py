"""
Smart Article Insights Generator
Phase 3: Full RAG Q&A (FAISS + all-MiniLM-L6-v2 + TinyLlama)
Phase 2: TinyLlama article summarization (unchanged)
"""

import os
import re
import functools
import traceback

# IMPORTANT: spaces must be imported BEFORE any CUDA-related packages
# (torch, faiss, transformers, sentence_transformers) to satisfy ZeroGPU.
import spaces

import numpy as np
import pandas as pd
import faiss
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()  # loads HF_TOKEN / FORCE_CPU from .env if present

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID     = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
CSV_PATH     = os.path.join(os.path.dirname(__file__), "Articles.csv")

# Chunking parameters (RecursiveCharacterTextSplitter style)
CHUNK_SIZE    = 500   # characters per chunk
CHUNK_OVERLAP = 50    # overlap between adjacent chunks

# RAG retrieval
TOP_K = 5   # number of chunks to retrieve

# Detect device once at import time
_FORCE_CPU = os.getenv("FORCE_CPU", "0") == "1"
DEVICE = "cpu" if _FORCE_CPU or not torch.cuda.is_available() else "cuda"
print(f"[INFO] Using device: {DEVICE}")


# ---------------------------------------------------------------------------
# ① TinyLlama — cached loader (Phase 2, unchanged)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def load_model():
    """
    Load TinyLlama tokenizer + model exactly once.
    Returns (tokenizer, model). Raises RuntimeError on failure.
    """
    print(f"[INFO] Loading LLM '{MODEL_ID}' on {DEVICE} …")
    try:
        hf_token = os.getenv("HF_TOKEN") or None

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, use_fast=True, token=hf_token
        )
        dtype = torch.float16 if DEVICE == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            token=hf_token,
        )
        model.to(DEVICE)
        model.eval()
        print("[INFO] LLM loaded successfully.")
        return tokenizer, model
    except Exception as exc:
        raise RuntimeError(f"Failed to load LLM: {exc}") from exc


# ---------------------------------------------------------------------------
# ② Sentence Transformer — cached loader (Phase 3)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def load_embedder():
    """
    Load all-MiniLM-L6-v2 once. Always runs on CPU (small model, fast enough).
    """
    print(f"[INFO] Loading embedder '{EMBED_MODEL}' …")
    embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
    print("[INFO] Embedder loaded.")
    return embedder


# ---------------------------------------------------------------------------
# ③ Articles.csv — loaded once at startup
# ---------------------------------------------------------------------------

def load_articles() -> dict[str, str]:
    """
    Return {title: content} dict from Articles.csv.
    Falls back gracefully if the file is missing.
    """
    try:
        df = pd.read_csv(CSV_PATH)
        return dict(zip(df["Title"].str.strip(), df["Article"].str.strip()))
    except Exception as exc:
        print(f"[WARNING] Could not load Articles.csv: {exc}")
        return {}

ARTICLES: dict[str, str] = load_articles()
ARTICLE_TITLES: list[str] = list(ARTICLES.keys())
print(f"[INFO] Loaded {len(ARTICLES)} articles from CSV.")


# ---------------------------------------------------------------------------
# ④ Recursive character text splitter
# ---------------------------------------------------------------------------

def recursive_split(text: str, chunk_size: int = CHUNK_SIZE,
                    overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks using sentence/word boundaries,
    mimicking LangChain's RecursiveCharacterTextSplitter logic.
    """
    separators = ["\n\n", "\n", ". ", " ", ""]

    def _split(text: str, seps: list[str]) -> list[str]:
        if len(text) <= chunk_size or not seps:
            return [text]
        
        sep = seps[0]
        next_seps = seps[1:]
        for i, s in enumerate(seps):
            if s == "" or s in text:
                sep = s
                next_seps = seps[i+1:]
                break
                
        parts = text.split(sep) if sep else list(text)
        
        chunks = []
        current_chunk = []
        current_len = 0
        
        for part in parts:
            if len(part) > chunk_size and next_seps:
                if current_chunk:
                    chunks.append(sep.join(current_chunk))
                    current_chunk = []
                    current_len = 0
                chunks.extend(_split(part, next_seps))
            else:
                part_len = len(part) + (len(sep) if current_chunk else 0)
                if current_len + part_len > chunk_size and current_chunk:
                    chunks.append(sep.join(current_chunk))
                    while current_chunk and (current_len > overlap or current_len + part_len > chunk_size):
                        popped = current_chunk.pop(0)
                        current_len -= len(popped) + (len(sep) if current_chunk else 0)
                
                current_chunk.append(part)
                current_len += len(part) + (len(sep) if len(current_chunk) > 1 else 0)
                
        if current_chunk:
            chunks.append(sep.join(current_chunk))
            
        return chunks

    return [c.strip() for c in _split(text, separators) if c.strip()]


# ---------------------------------------------------------------------------
# ⑤ FAISS index builder
# ---------------------------------------------------------------------------

def build_faiss_index(chunks: list[str]):
    """
    Embed `chunks` with all-MiniLM-L6-v2, store in a FAISS IndexFlatL2.
    Returns (index, embeddings_array).
    """
    embedder = load_embedder()
    embeddings = embedder.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    embeddings = embeddings.astype(np.float32)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings


# ---------------------------------------------------------------------------
# ⑥ RAG pipeline helpers
# ---------------------------------------------------------------------------

def retrieve_chunks(question: str, chunks: list[str],
                    index: faiss.IndexFlatL2, top_k: int = TOP_K) -> list[str]:
    """
    Embed `question`, query FAISS, return top-k chunk strings.
    """
    embedder = load_embedder()
    q_vec = embedder.encode([question], convert_to_numpy=True,
                            normalize_embeddings=True).astype(np.float32)
    distances, indices = index.search(q_vec, top_k)
    return [chunks[i] for i in indices[0] if 0 <= i < len(chunks)]


def _trim_to_last_sentence(text: str) -> str:
    """
    Clip `text` at the last sentence-ending punctuation (. ! ?).
    Removes trailing incomplete numbered-list markers (e.g. '4.').
    Guarantees the returned string ends with a complete sentence.
    """
    text = text.strip()
    
    while True:
        last = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
        if last == -1:
            return text
            
        trimmed = text[: last + 1].strip()
        
        # Check if the trimmed text ends with an incomplete list marker (e.g., "4.")
        match = re.search(r'(?:^|\n)\s*\d+\.$', trimmed)
        if match:
            # It's an incomplete list marker. Remove it entirely and trim again.
            text = trimmed[:match.start()].strip()
            if not text:
                return ""
            continue
            
        return trimmed


@spaces.GPU(duration=60)
def generate_rag_answer(question: str, context: str) -> str:
    """
    Feed (context, question) to TinyLlama using a compact prompt.
    Decodes only newly generated tokens.
    """
    try:
        tokenizer, model = load_model()
    except RuntimeError as exc:
        return f"❌ Model loading failed:\n{exc}"

    FALLBACK = "I don't know based on the provided context."

    # Strict grounding prompt: CONTEXT / QUESTION / INSTRUCTIONS / ANSWER
    system_msg = (
        "You are a strict factual assistant. "
        "You MUST answer using ONLY the information explicitly present in the CONTEXT. "
        "You MUST NOT use any prior knowledge, training data, or general facts. "
        "You MUST NOT guess, infer, or add any information that is not stated word-for-word in the CONTEXT. "
        "You MUST NOT include any facts, statistics, names, dates, salaries, or sources "
        "that do not appear verbatim in the CONTEXT. "
        f'If the CONTEXT does not explicitly contain the answer, output EXACTLY this sentence and nothing else: "{FALLBACK}"'
    )
    user_msg = (
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{question}\n\n"
        "INSTRUCTIONS:\n"
        "Answer ONLY using the CONTEXT above.\n"
        f'If the CONTEXT does not contain the answer, output exactly: "{FALLBACK}"\n\n'
        "ANSWER:"
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_msg},
    ]

    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1900,
        ).to(DEVICE)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=140,
                do_sample=False,
                repetition_penalty=1.15,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][input_len:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # ── Post-generation safety check ─────────────────────────────────────
        # If the model's answer already IS the fallback, return it untrimmed.
        if answer == FALLBACK:
            return FALLBACK

        # Pass 1: keyword self-report — model explicitly says it doesn't know.
        _cannot_answer_phrases = [
            "i don't know",
            "i do not know",
            "not mentioned in the context",
            "not provided in the context",
            "context does not",
            "context doesn't",
            "not in the context",
            "cannot answer",
            "can't answer",
            "unable to answer",
        ]
        answer_lower = answer.lower()
        if any(phrase in answer_lower for phrase in _cannot_answer_phrases):
            return FALLBACK

        # Pass 2: Key-noun grounding check.
        # Catch hallucinations where the model generates content from entirely
        # outside the context (e.g. salary figures, external sources, dates).
        # Strategy: extract meaningful nouns from the answer that are NOT in the
        # context at all. If > 40% of meaningful answer words are entirely absent,
        # it's a hallucination.
        _STOPWORDS = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "shall", "can", "of", "in", "on", "at",
            "to", "for", "with", "by", "from", "that", "this", "these", "those",
            "it", "its", "and", "or", "but", "not", "as", "such", "which", "also",
            "their", "they", "them", "there", "than", "then", "so", "if", "all",
            "some", "any", "more", "most", "each", "other", "into", "about",
            "up", "out", "no", "how", "what", "when", "where", "who", "i",
            "we", "you", "he", "she", "his", "her", "our", "your", "my",
            # Common connecting words likely in paraphrased answers:
            "include", "includes", "including", "allow", "allows", "help",
            "helps", "provide", "provides", "enable", "enables", "use",
            "used", "using", "make", "makes", "improve", "improves",
            "overall", "additionally", "furthermore", "however",
        }
        context_lower = context.lower()

        def _meaningful_words(text: str) -> set:
            words = re.findall(r"[a-z]+", text.lower())
            return {w for w in words if w not in _STOPWORDS and len(w) > 3}

        answer_words  = _meaningful_words(answer)
        # A word is "grounded" if it appears anywhere in the raw context string
        grounded = {w for w in answer_words if w in context_lower}
        ungrounded = answer_words - grounded

        if answer_words:
            ungrounded_ratio = len(ungrounded) / len(answer_words)
            if ungrounded_ratio > 0.60:   # > 60% of key words absent from context
                return FALLBACK

        # Normal path: trim to last complete sentence and return.
        trimmed = _trim_to_last_sentence(answer)
        return trimmed if trimmed else "⚠️ Model returned an empty response."

    except Exception:
        return f"❌ Inference error:\n{traceback.format_exc()}"


# ---------------------------------------------------------------------------
# ⑦ Per-article FAISS cache  (rebuilt only when article selection changes)
# ---------------------------------------------------------------------------

# Holds: {article_title: (chunks, faiss_index)}
_article_index_cache: dict[str, tuple[list[str], faiss.IndexFlatL2]] = {}


def get_article_index(article_title: str) -> tuple[list[str], faiss.IndexFlatL2]:
    """
    Return (chunks, faiss_index) for `article_title`.
    Builds and caches on first access; returns cached result on subsequent calls.
    """
    if article_title not in _article_index_cache:
        print(f"[INFO] Building index for article: '{article_title}'")
        content = ARTICLES[article_title]
        chunks = recursive_split(content)
        index, _ = build_faiss_index(chunks)
        _article_index_cache[article_title] = (chunks, index)
    return _article_index_cache[article_title]


# ---------------------------------------------------------------------------
# ⑧ Gradio callback — Tab 2
# ---------------------------------------------------------------------------

def answer_question(article_title: str, question: str):
    """
    RAG Q&A — per-question work is ONLY: embed question → FAISS search → generate.
    Chunking and indexing happen once per article (cached).
    Returns (answer, retrieved_context_display).
    """
    question = question.strip()
    if not question:
        return "⚠️ Please enter a question.", ""

    if not article_title or article_title not in ARTICLES:
        return "⚠️ Please select a valid article.", ""

    # ── Get cached chunks + index (or build once) ────────────────────────────
    try:
        chunks, index = get_article_index(article_title)
        if not chunks:
            return "⚠️ Article content is empty or could not be split.", ""
    except Exception:
        return f"❌ Indexing error:\n{traceback.format_exc()}", ""

    # ── Retrieve — question embedding + FAISS search only ────────────────────
    try:
        top_chunks = retrieve_chunks(question, chunks, index, top_k=TOP_K)
    except Exception:
        return f"❌ Retrieval error:\n{traceback.format_exc()}", ""

    context = "\n\n".join(top_chunks)

    # ── Generate answer ──────────────────────────────────────────────────────
    answer = generate_rag_answer(question, context)

    # ── Format retrieved context for display ─────────────────────────────────
    context_display = "\n\n---\n\n".join(
        f"[Chunk {i+1}]\n{c}" for i, c in enumerate(top_chunks)
    )

    return answer, context_display


# ---------------------------------------------------------------------------
# ⑧ Tab 1 — Article Summarization (Phase 2, completely unchanged)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful assistant that produces clear, concise summaries. "
    "Summarize the article provided by the user in 3-5 complete sentences. "
    "Cover the key points and end with a complete sentence."
)


@spaces.GPU(duration=60)
def summarize_article(article_text: str) -> str:
    """
    Use TinyLlama to generate a concise summary of the given article.
    Only newly generated tokens are decoded (input prompt is excluded).
    """
    article_text = article_text.strip()
    if not article_text:
        return "⚠️ Please paste an article before generating a summary."

    try:
        tokenizer, model = load_model()
    except RuntimeError as exc:
        return f"❌ Model loading failed:\n{exc}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Article:\n\n{article_text}"},
    ]

    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1800
        ).to(DEVICE)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=180,      # allows complete, uncut summary
                do_sample=False,
                repetition_penalty=1.15,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][input_len:]
        summary = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        # Trim to the last complete sentence so the output never ends mid-phrase
        summary = _trim_to_last_sentence(summary)
        return summary if summary else "⚠️ Model returned an empty response. Try a shorter article."

    except Exception:
        return f"❌ Inference error:\n{traceback.format_exc()}"


# ---------------------------------------------------------------------------
# ⑨ Gradio UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Smart Article Insights Generator") as demo:

        gr.Markdown(
            """
            # 📰 Smart Article Insights Generator
            _Powered by TinyLlama · all-MiniLM-L6-v2 · FAISS_
            """
        )

        # ── Tab 1: Summarize ──────────────────────────────────────────────
        with gr.Tab("📝 Summarize Article"):
            gr.Markdown(
                "Paste any article below and click **Generate Summary**. "
                f"_(Running on **{DEVICE.upper()}**)_"
            )
            with gr.Row():
                with gr.Column(scale=2):
                    article_input = gr.Textbox(
                        label="Article Text",
                        placeholder="Paste your article here…",
                        lines=18,
                        max_lines=30,
                        elem_id="article_input",
                    )
                    summarize_btn = gr.Button(
                        "⚡ Generate Summary",
                        variant="primary",
                        elem_id="summarize_btn",
                    )
                with gr.Column(scale=1):
                    summary_output = gr.Textbox(
                        label="Summary",
                        placeholder="Summary will appear here…",
                        lines=18,
                        interactive=False,
                        elem_id="summary_output",
                    )
            summarize_btn.click(
                fn=summarize_article,
                inputs=[article_input],
                outputs=[summary_output],
            )

        # ── Tab 2: RAG Q&A ────────────────────────────────────────────────
        with gr.Tab("🔍 Ask Questions (RAG)"):
            gr.Markdown(
                "Select an article, type your question, and click **Ask Question**. "
                "Answers are grounded strictly in the article content."
            )

            article_selector = gr.Dropdown(
                choices=ARTICLE_TITLES,
                label="Select Article",
                value=ARTICLE_TITLES[0] if ARTICLE_TITLES else None,
                elem_id="article_selector",
            )

            question_input = gr.Textbox(
                label="Your Question",
                placeholder="e.g. What are the main challenges in AI healthcare?",
                lines=3,
                elem_id="question_input",
            )

            ask_btn = gr.Button(
                "🔎 Ask Question",
                variant="primary",
                elem_id="ask_btn",
            )

            answer_output = gr.Textbox(
                label="Answer",
                lines=7,
                interactive=False,
                elem_id="answer_output",
            )

            with gr.Accordion("📄 Retrieved Context (debug)", open=False):
                context_output = gr.Textbox(
                    label="Top Retrieved Chunks",
                    lines=10,
                    interactive=False,
                    elem_id="context_output",
                )

            ask_btn.click(
                fn=answer_question,
                inputs=[article_selector, question_input],
                outputs=[answer_output, context_output],
            )

        gr.Markdown(
            """
            ---
            > **Phase 3** – Full RAG pipeline active (FAISS · all-MiniLM-L6-v2 · TinyLlama).
            """
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        ssr_mode=False,
    )
