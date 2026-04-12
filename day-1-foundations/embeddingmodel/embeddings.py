
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=64)

vector = embeddings.embed_query("you are a helpful assistant")

print(vector)