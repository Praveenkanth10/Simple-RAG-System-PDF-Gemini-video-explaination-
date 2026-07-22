"# Simple RAG System (PDF + Gemini)
Upload a PDF, ask questions about it. Uses Gemini for both embeddings and answer generation — no other AI service needed.

## 1. Install dependencies
"""

!pip install -q google-generativeai pypdf

"""## 2. Set your Gemini API key
Get a free key at https://aistudio.google.com/apikey

Add it in Colab's Secrets manager (key icon 🔑 in the left sidebar) as `GEMINI_API_KEY`, then run this cell.
"""

import google.generativeai as genai
from google.colab import userdata

genai.configure(api_key=userdata.get('GEMINI_API_KEY'))

"""## 3. Upload your PDF"""

from google.colab import files

uploaded = files.upload()
pdf_path = list(uploaded.keys())[0]
print(f"Uploaded: {pdf_path}")

"""## 4. Extract text from the PDF"""

from pypdf import PdfReader

reader = PdfReader(pdf_path)
full_text = ""
for page in reader.pages:
    full_text += page.extract_text() + "\n"

print(f"Extracted {len(full_text)} characters from {len(reader.pages)} pages.")

"""## 5. Chunk the text
Split into overlapping word chunks so context isn't cut off at boundaries.
"""

def chunk_text(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + chunk_size]))
        start += chunk_size - overlap
    return chunks

chunks = chunk_text(full_text)
print(f"Split into {len(chunks)} chunks.")

"""## Bonus: Watch the chunking process as a video
Generates an actual `.mp4` showing the sliding window move through your PDF's text, chunk by chunk, with the overlap visible — useful for explaining the process to freshers.
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import textwrap
from IPython.display import Video

def animate_chunking(text, chunk_size=8, overlap=2, max_words=48, filename="chunking_process.mp4"):
    words = text.split()[:max_words]  # keep the demo short so it stays watchable

    starts = []
    s = 0
    while s < len(words):
        starts.append(s)
        if s + chunk_size >= len(words):
            break
        s += chunk_size - overlap

    words_per_line = 8
    lines = [words[i:i + words_per_line] for i in range(0, len(words), words_per_line)]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    word_positions = []
    line_height = 15 / max(len(lines), 1)
    idx = 0
    for row, line in enumerate(lines):
        y = 0.95 - row * line_height
        x = 0.02
        for w in line:
            word_positions.append((idx, x, y, w))
            x += (len(w) + 1) * 0.017
            idx += 1

    highlight_patches = []
    text_objs = [ax.text(x, y, w, fontsize=11, va="center", ha="left", zorder=3)
                 for idx, x, y, w in word_positions]

    title = ax.text(0.5, 1.02, "", fontsize=13, ha="center", weight="bold")
    chunk_box = ax.text(0.02, 0.02, "", fontsize=10, va="bottom", ha="left", family="monospace")

    def update(frame):
        for p in highlight_patches:
            p.remove()
        highlight_patches.clear()

        if frame >= len(starts):
            title.set_text(f"Done — {len(starts)} chunks created")
            chunk_box.set_text("")
            return text_objs

        start = starts[frame]
        end = min(start + chunk_size, len(words))
        title.set_text(f"Chunk {frame + 1} of {len(starts)}  (words {start}-{end - 1})")

        for idx, x, y, w in word_positions:
            if start <= idx < end:
                rect = plt.Rectangle((x - 0.005, y - 0.03), (len(w) + 1) * 0.017, 0.06,
                                      facecolor="#85B7EB", alpha=0.6, zorder=1)
                ax.add_patch(rect)
                highlight_patches.append(rect)

        chunk_text = " ".join(words[start:end])
        wrapped = textwrap.fill(f"Chunk {frame + 1}: {chunk_text}", width=90)
        chunk_box.set_text(wrapped)

        return text_objs + highlight_patches

    ani = animation.FuncAnimation(fig, update, frames=len(starts) + 1, interval=1200, blit=False)
    ani.save(filename, writer="ffmpeg", fps=1)
    plt.close(fig)
    return filename

# Uses the real text extracted from your uploaded PDF (from step 4)
video_path = animate_chunking(full_text, chunk_size=8, overlap=2, max_words=48)
Video(video_path, embed=True, width=700)

"""## 6. Embed each chunk with Gemini
Turns each chunk of text into a vector that captures its meaning.
"""

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
                time.sleep(delay)  # stay under the per-minute quota
        except ResourceExhausted as e:
            print("Rate limited — waiting 65s before retrying this batch...")
            time.sleep(65)
            # loop retries the same batch, doesn't advance i

    return np.array(all_embeddings)

chunk_embeddings = embed(chunks)
print(f"Embedded {len(chunk_embeddings)} chunks. Shape: {chunk_embeddings.shape}")

"""## 7. Ask a question — retrieve the most relevant chunks
Embeds your question, then finds the chunks most similar to it using cosine similarity.
"""

def retrieve(query, top_k=3):
    query_embedding = embed([query], task_type="retrieval_query")[0]

    # cosine similarity between the query and every chunk
    scores = chunk_embeddings @ query_embedding / (
        np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [chunks[i] for i in top_indices]

"""## 8. Generate an answer, grounded in the retrieved chunks"""

model = genai.GenerativeModel("gemini-flash-latest")

def ask(query, top_k=3):
    relevant_chunks = retrieve(query, top_k=top_k)
    context = "\n\n".join(relevant_chunks)

    prompt = f"""Answer the question using ONLY the context below.
If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {query}

Answer:"""

    response = model.generate_content(prompt)
    return response.text

"""## 9. Try it"""

print(ask("What is this document about?"))

# Ask anything else about your PDF
print(ask("what is machine learning"))
