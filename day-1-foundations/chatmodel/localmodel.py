from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={
        "repetition_penalty": 1.2, 
        "max_new_tokens": 512, 
        "temperature": 0.6,
        "do_sample": True
    },
)

chat_model = ChatHuggingFace(llm=llm)

print("Model loaded. Generating DNS explanation...")

result = chat_model.invoke("explain me about DNS and how it works")
print(result.content)