import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import requests
import json

def create_embeddings(text_list):
    r = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "bge-m3",
            "input": text_list
        }
    )
    embedding = r.json()['embeddings']
    return embedding


def inferencef(prompt):
    with requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2",   # or "deepseek-r1"
            "prompt": prompt,
            "stream": True
        },
        stream=True
    ) as r:
        for line in r.iter_lines():
            if line:
                data = json.loads(line.decode("utf-8"))
                if "response" in data:
                    print(data["response"], end="", flush=True)
                if data.get("done"):
                    break


df = joblib.load("embeddings.joblib")

incoming_query = input("Ask a Question: ")
question_embedding = create_embeddings([incoming_query])[0]

# Compute similarity
similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()

# Get top 3 most relevant chunks
top_result = 3
max_indx = similarities.argsort()[::-1][0:top_result]
new_df = df.loc[max_indx]

# Build compact and efficient prompt
prompt = f"""
You are a helpful assistant for the Sigma Web Development Course.

Here are subtitle chunks with title, video number, start, end, and text:
{new_df[['title','number','start','end','text']].to_json(orient='records')}

User question: "{incoming_query}"

Answer simply and naturally:
- Mention which video and timestamp cover this topic.
- Briefly say what is explained there.
If not related to the course, say you can only answer course-related questions.
"""

with open("prompt.txt", "w") as f:
    f.write(prompt)
print("\nThinking...\n")

# Stream output in real-time
inferencef(prompt)
print("\n\nDone.")
