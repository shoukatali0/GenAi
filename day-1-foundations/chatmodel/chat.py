import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
import os
print(f"API Key found: {os.getenv('MISTRAL_API_KEY') is not None}")
load_dotenv("../.env")  # Load environment variables from .env file


from dotenv import load_dotenv

# model = init_chat_model(model = "mistral-small-2603", temprature=0.9)

model = init_chat_model(
    model="mistral-small-latest", # using 'latest' is safer than specific dates
    model_provider="mistralai",
    temperature=0,
    max_tokens=20
)

# #model = init_chat_model(
#     "llama-3.3-70b-versatile", # Or your specific Llama 4 string if supported
#     model_provider="groq"
# #)#
responce = model.invoke("write a poem about love that i can never have in my life, make it sad and beautiful")


print(responce.content)