from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv
load_dotenv()


embedding = HuggingFaceEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
)


vector = embedding.embed_query("you are a helpful assistant")

print(vector)