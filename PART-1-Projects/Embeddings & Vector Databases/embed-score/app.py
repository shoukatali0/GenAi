from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


#loading local embedding model

MODEL_PATH = "sentence-transformers/all-MiniLM-L6-v2"

print("Loading SentenceTransformer model...")
model = SentenceTransformer(MODEL_PATH)


print("Model loaded successfully")

sentences = [
    "Linux is an open-source operating system.",
    "Ubuntu is a popular Linux distribution.",
    "AWS EC2 provides virtual servers in the cloud.",
    "Amazon S3 stores files and objects.",
    "Docker packages applications into containers.",
    "Kubernetes manages containerized applications.",
    "Customers can return products within thirty days.",
    "Refunds are processed after the returned item is inspected.",
    "Today's weather in Tokyo is sunny.",
    "Rain is expected tomorrow in Mumbai.",
    "Pasta is usually cooked in boiling water.",
    "Pizza is traditionally baked in an oven.",
    "Python is a popular programming language.",
    "Machine learning enables computers to learn from data.",
    "Embeddings convert text into numerical vectors."
]


#genarting embedding for the sentences

sentence_embeddings = model.encode(sentences)


print("Embeddings generated successfully")

print(sentence_embeddings.shape)

print(f"Generated {len(sentence_embeddings)} embeddings.\n")

query = input("Enter your query: ")

query_embedding = model.encode([query])


#calculate cosine similarity

similarities = cosine_similarity(query_embedding, sentence_embeddings)[0]


#rank results

top_indices = np.argsort(similarities)[::-1][:3]

# print("Similarities:", similarities)

print(f"\nTop 3 most similar sentences to: '{query}'")

for rank, i in enumerate(top_indices, start=1):
    score = similarities[i]
    print(f"{rank}.) (score: {score:.4f}) {sentences[i]}")

    