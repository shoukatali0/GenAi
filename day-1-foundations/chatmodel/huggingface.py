from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv("../.env") 

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    temperature=0.6

)

model = ChatHuggingFace(llm=llm)


response = model.invoke("tell me about the history of the world in 100 words")

print(response.content)