import numpy as np
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from questions import questions
from chunks import chunks

# 1. Load model
print("Loading model...")
model = SentenceTransformer('intfloat/multilingual-e5-large')
print("Model loaded ✅\n")

# 2. Embed everything
# multilingual-e5 requires a prefix for queries and passages
print("Embedding questions...")
question_embeddings = model.encode(
    ["query: " + q for q in questions],
    normalize_embeddings=True,
    show_progress_bar=True
)

print("\nEmbedding chunks...")
chunk_embeddings = model.encode(
    ["passage: " + c for c in chunks],
    normalize_embeddings=True,
    show_progress_bar=True
)

# 3. Compute cosine similarity
similarity_matrix = cosine_similarity(question_embeddings, chunk_embeddings)

# 4. For each question, retrieve top 3 chunks
print("\n========== RESULTS ==========\n")
results = []

for i, question in enumerate(questions):
    scores = similarity_matrix[i]
    top3_indices = np.argsort(scores)[::-1][:3]
    top3 = [(idx, round(float(scores[idx]), 4)) for idx in top3_indices]
    
    print(f"Q{i+1}: {question}")
    for rank, (chunk_idx, score) in enumerate(top3, 1):
        print(f"  Top {rank} (score: {score}) → Chunk {chunk_idx+1}: {chunks[chunk_idx][:80]}...")
    print()

    results.append({
        "question_id": f"Q{i+1}",
        "question": question,
        "top3": [
            {"chunk_id": int(idx+1), "score": score, "chunk": chunks[idx]}
            for idx, score in top3
        ]
    })

# 5. Save raw results to JSON
with open("results/raw_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("Raw results saved to results/raw_results.json ✅")