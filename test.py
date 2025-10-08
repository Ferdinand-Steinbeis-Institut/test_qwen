# paper_reader.py
# pip install openai pypdf tiktoken
import os, sys, math
from typing import List
from pypdf import PdfReader
import tiktoken
from openai import OpenAI

MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")  # or "gpt-5" if available to you
MAX_TOKENS_PER_CHUNK = 1400  # safe margin for 4-8k+ context models; adjust per model
SUMMARY_TARGET_TOKENS = 400

client = OpenAI()  # expects OPENAI_API_KEY in env

def extract_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")  # skip unreadable pages
    return "\n\n".join(pages)

def chunk_text(text: str, model: str, max_tokens: int) -> List[str]:
    # token-aware splitting
    enc = tiktoken.encoding_for_model(model) if model else tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    chunks = []
    step = max_tokens
    for start in range(0, len(tokens), step):
        piece = tokens[start:start+max_tokens]
        chunks.append(enc.decode(piece))
    return chunks

def llm(prompt: str, system: str = None, temperature: float = 0.2) -> str:
    # Responses API: simple text I/O
    parts = []
    if system:
        parts.append({"role": "system", "content": system})
    parts.append({"role": "user", "content": prompt})
    resp = client.responses.create(model=MODEL, input=parts, temperature=temperature)
    return resp.output_text  # unified text accessor

def summarize_chunk(chunk: str, idx: int, total: int) -> str:
    return llm(
        f"""You are a scientific paper reading assistant.
Read this chunk {idx}/{total} and produce:
- 2–3 bullet point takeaways (plain text bullets)
- Important definitions/equations (if any)
- Citations/section refs mentioned (preserve identifiers)
Chunk:
\"\"\"{chunk}\"\"\""""
    )

def merge_summaries(summaries: List[str], paper_title: str) -> str:
    joined = "\n\n---\n\n".join(summaries)
    return llm(
        f"""You are compiling a concise, technically accurate paper brief.

Paper: {paper_title}

Given the chunk notes below, produce:
1) 6–10 bullet Executive Summary (non-redundant)
2) Key Concepts & Definitions
3) Methods (short)
4) Results (with numbers where present)
5) Limitations
6) Suggested Follow-ups / Questions to ask the authors

Chunk notes:
\"\"\"{joined}\"\"\""""
    , temperature=0.1)

def answer_question(question: str, summaries: List[str]) -> str:
    context = "\n\n---\n\n".join(summaries[-8:])  # last few chunks often contain results/appendix
    return llm(
        f"""Use only the provided notes to answer succinctly and cite sections/figures when possible.
If unknown, say so and suggest where to look next.
Context:
\"\"\"{context}\"\"\"

Q: {question}
A:"""
    )

def main():
    if len(sys.argv) < 2:
        print("Usage: python paper_reader.py path/to/paper.pdf [\"Optional Paper Title\"]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    paper_title = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(pdf_path)

    print("Extracting text…")
    text = extract_text(pdf_path)

    print("Chunking…")
    chunks = chunk_text(text, MODEL, MAX_TOKENS_PER_CHUNK)
    total = len(chunks)

    print(f"Summarizing {total} chunks…")
    per_chunk = []
    for i, ch in enumerate(chunks, start=1):
        note = summarize_chunk(ch, i, total)
        per_chunk.append(note)
        print(f"✓ chunk {i}/{total}")

    print("Merging into a paper brief…")
    brief = merge_summaries(per_chunk, paper_title)

    # Save artifacts
    out_dir = os.path.splitext(pdf_path)[0] + "_agent"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "chunk_notes.txt"), "w", encoding="utf-8") as f:
        f.write("\n\n====\n\n".join(per_chunk))
    with open(os.path.join(out_dir, "paper_brief.md"), "w", encoding="utf-8") as f:
        f.write(brief)

    print("\n=== PAPER BRIEF ===\n")
    print(brief)
    print("\nAsk a question (or press Enter to exit):")
    while True:
        q = input("> ").strip()
        if not q:
            break
        print(answer_question(q, per_chunk), "\n")

if __name__ == "__main__":
    main()
