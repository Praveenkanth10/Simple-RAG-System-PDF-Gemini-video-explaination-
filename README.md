# RAG-Based PDF Question-Answering System

A Retrieval-Augmented Generation (RAG) pipeline that lets you upload any PDF and ask natural-language questions about it — answers are grounded in the document itself, not the model's memorized training data.

Built with **Python**, **Google Gemini API**, and **Google Colab**.

## How it works

```
PDF → Extract Text → Chunk → Embed → Store Vectors
                                          │
User Question → Embed Query → Retrieve Top-K Similar Chunks
                                          │
                    Chunks + Question → Gemini → Grounded Answer
```

1. **Extract** — pulls raw text out of the uploaded PDF using `pypdf`
2. **Chunk** — splits the text into overlapping word chunks (200 words, 50-word overlap) so ideas near chunk boundaries aren't lost
3. **Embed** — converts each chunk into a semantic vector using Gemini's `gemini-embedding-001` model
4. **Store** — keeps all chunk vectors in memory for fast lookup
5. **Retrieve** — embeds the user's question and finds the most relevant chunks using cosine similarity
6. **Generate** — feeds the retrieved chunks + question to `gemini-3.5-flash`, instructed to answer only from the provided context (reduces hallucination)

## Demo

> Add a screenshot or GIF here showing a question and the answer your system returned.

## Tech Stack

- **Python** — core pipeline logic
- **Google Gemini API** (`gemini-embedding-001`, `gemini-3.5-flash`) — embeddings & generation
- **google-generativeai** — Gemini Python SDK
- **pypdf** — PDF text extraction
- **NumPy** — vector math / cosine similarity
- **Google Colab** — development & deployment environment

## Getting Started

### Run in Google Colab (recommended, zero setup)

1. Open [`rag_pdf_gemini.ipynb`](./rag_pdf_gemini.ipynb) directly in Colab
2. Get a free Gemini API key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
3. Add it to Colab's Secrets manager (🔑 icon, left sidebar) as `GEMINI_API_KEY`
4. Run all cells — you'll be prompted to upload a PDF, then can ask questions about it

### Run locally

```bash
git clone https://github.com/<your-username>/rag-pdf-qa-gemini.git
cd rag-pdf-qa-gemini
pip install -r requirements.txt
export GEMINI_API_KEY="your-key-here"
python rag_system.py
```

## Project Structure

```
rag-pdf-qa-gemini/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── rag_pdf_gemini.ipynb               # Main Colab notebook (PDF upload + Q&A)
├── rag_tutorial_video_generator.ipynb # Generates a narrated explainer video of the pipeline
└── LICENSE
```

## Troubleshooting

Real errors encountered while building this, and how they were fixed — useful if you hit the same ones.

### `404: models/text-embedding-004 is not found`

Google retired `text-embedding-004` in January 2026. Use the current model instead:

```python
result = genai.embed_content(
    model="models/gemini-embedding-001",  # not text-embedding-004
    content=texts,
    task_type="retrieval_document",
)
```

### `404: models/gemini-2.5-flash is no longer available to new users`

`gemini-2.5-flash` was deprecated for new API users in favor of newer Flash models. Use:

```python
model = genai.GenerativeModel("gemini-3.5-flash")
```

If this changes again in the future, list what's actually available on your account:

```python
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)
```

### `429: Quota exceeded for embed_content_free_tier_requests`

The free tier allows a limited number of embedding requests per minute (roughly 100). The quota counts **each chunk in a batch as a separate request**, not each API call — so sending 90 chunks in one `embed_content` call uses ~90 of that allowance instantly.

Fix: batch in smaller groups, pace requests with a delay, and auto-retry on rate limit:

```python
import time
import numpy as np
from google.api_core.exceptions import ResourceExhausted

def embed(texts, task_type="retrieval_document", batch_size=40, delay=65):
    all_embeddings = []
    i = 0
    while i < len(texts):
        batch = texts[i:i + batch_size]
        try:
            result = genai.embed_content(
                model="models/gemini-embedding-001",
                content=batch,
                task_type=task_type,
            )
            all_embeddings.extend(result["embedding"])
            i += batch_size
            print(f"Embedded {min(i, len(texts))}/{len(texts)} chunks...")
            if i < len(texts):
                time.sleep(delay)
        except ResourceExhausted:
            print("Rate limited — waiting 65s before retrying this batch...")
            time.sleep(65)

    return np.array(all_embeddings)
```

**Faster alternatives:**
- Increase `chunk_size` in the chunking step (e.g. 200 → 400 words) to generate fewer, larger chunks and cut the number of API calls roughly in half
- Enable billing on your Google AI Studio project — embedding costs a fraction of a cent per document and unlocks a much higher rate limit, removing the need for batching/delays entirely

## Why RAG?

Large language models have a training cutoff and no knowledge of private or recent documents. RAG solves this by retrieving relevant information from a document at question time and grounding the model's answer in that retrieved context — the same core pattern used in production tools like "chat with your docs" assistants and internal knowledge-base search.

## Future Improvements

- [ ] Support multiple documents in a single session
- [ ] Persist embeddings to disk / a vector database instead of in-memory storage
- [ ] Add a simple web UI (Streamlit/Gradio)
- [ ] Support batching for larger PDFs to avoid embedding rate limits

## License

MIT — feel free to use this for learning or as a starting point for your own project.
