
# import os
# from dotenv import load_dotenv
# from langchain.chat_models import init_chat_model
# import os
# print(f"API Key found: {os.getenv('MISTRAL_API_KEY') is not None}")
# load_dotenv("../.env")  # Load environment variables from .env file


# from dotenv import load_dotenv

# # model = init_chat_model(model = "mistral-small-2603", temprature=0.9)

# model = init_chat_model(
#     model="mistral-small-latest", # using 'latest' is safer than specific dates
#     model_provider="mistralai",
#     temperature=0,
#     max_tokens=256


from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import  AIMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv
load_dotenv("../.env") 

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    temperature=0.9,
    max_tokens=1024

)

model = ChatHuggingFace(llm=llm)

print("select your AI mode: ")
print("1. phylosopher bot")
print("2. comedian bot")
print("3. motivational bot")
print("4. financial advisor bot")


choice = input("Enter your choice (1-4): ")

if choice == "1":
    mode = "You are a wise philosopher bot. You provide deep insights and thoughtful advice on life's big questions."
elif choice == "2":
    mode = "You are a comedian bot. You tell jokes and funny stories to make people laugh."   
elif choice == "3":
    mode = "You are a motivational bot. You inspire and encourage people to achieve their goals and overcome challenges." 
elif choice == "4":  
    mode = "You are a financial advisor bot. You provide expert advice on personal finance, investments, and money management."

messages = [
 SystemMessage(content=mode)


]  # Initialize an empty list to store  the conversation history

while True:
    print("--------------type 0 for exit----------------")
    prompt = input("YOU: ")

     # Add user message to the conversation history
    messages.append(HumanMessage(content=prompt ))
    if prompt == "0":
        print("Exiting the chat. Goodbye!")
        break
    responce = model.invoke(messages)  # Get the model's response based on the conversation history
    messages.append(AIMessage(content=responce.content)) # Add user message to the conversation history
    responce = model.invoke(messages)  # Get the model's response based on the conversation history
    messages.append({"role": "assistant", "content": responce.content})

    print("bot :",responce.content)

print(messages)

