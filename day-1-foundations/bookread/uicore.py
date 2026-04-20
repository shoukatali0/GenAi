from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
import os

load_dotenv()

app = Flask(__name__, static_folder=".")
CORS(app)

model = ChatMistralAI(model="mistral-small-2603")

prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are an expert literary analyst and intelligent information extractor.

Your task is to analyze the given paragraph and extract structured book information.

---------------------
INPUT PARAGRAPH:
{input_text}
---------------------

INSTRUCTIONS:

1. Extract information directly if available.
2. If information is missing, intelligently infer it based on context and general knowledge.
3. Avoid incorrect guesses — only infer when reasonably confident.
4. Keep output clean and structured.
5. Do NOT leave important fields empty.

---------------------
OUTPUT FORMAT:

Title: 
Author: 
Genre: 
Sub-Genre: 
Publication Year: 
Publisher: 
Language: 
Pages: 
Series (if any): 

Main Characters: 
Setting (Time & Place): 

Themes: 
Keywords: 

Writing Style: 
Tone: 
Target Audience: 

Notable Awards (if any): 

---------------------

Short Summary (2–3 lines):

---------------------

Detailed Summary (5–8 lines):

---------------------

Key Takeaways:
- 
- 
- 
- 

---------------------

IMPORTANT:
- Prefer intelligent completion over "Not mentioned"
- Keep answers realistic and factually aligned
- Do not hallucinate very specific unknown data (like ISBN)
"""),
    ("human", "extract the information from the following paragraph and provide a summary as per the instructions above:")
])


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    input_text = data.get("text", "").strip()

    if not input_text:
        return jsonify({"error": "No text provided"}), 400

    try:
        final_prompt = prompt.invoke({"input_text": input_text})
        response = model.invoke(final_prompt)
        return jsonify({"result": response.content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)