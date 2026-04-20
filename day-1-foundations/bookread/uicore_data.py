import streamlit as st

# --- your existing imports ---
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Optional, List
from langchain_core.output_parsers import PydanticOutputParser 
from langchain_mistralai import ChatMistralAI

# --- load env ---
load_dotenv()

# --- model ---
model = ChatMistralAI(model="mistral-small-2603")

# --- schema ---
class BookInfo(BaseModel):
    title: str = Field(description="The full title of the book")
    author: str = Field(description="The name of the author")
    release_year: Optional[int] = Field(description="The year the book was published")
    genre: List[str] = Field(description="List of genres")
    main_character: List[str] = Field(description="List of main characters")
    rating: Optional[float] = Field(description="The book's rating out of 10")
    summary: str = Field(description="A brief summary of the content")

parser = PydanticOutputParser(pydantic_object=BookInfo)

# --- prompt ---
prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are an expert literary analyst and intelligent information extractor.
{follow_instructions}
"""),
    ("human", "{input_text}"),
])

# ================= UI =================

st.set_page_config(page_title="📚 Book Extractor AI", layout="wide")

st.title("📚 Book Information Extractor")
st.markdown("Paste any book paragraph → get structured insights instantly")

# input box
para = st.text_area("Enter book description:", height=200)

# button
if st.button("Extract Info 🚀"):
    if para.strip() == "":
        st.warning("Please enter some text")
    else:
        with st.spinner("Analyzing..."):
            final_prompt = prompt.invoke({
                "input_text": para,
                "follow_instructions": parser.get_format_instructions()
            })

            responce = model.invoke(final_prompt)

        st.success("Done!")

        # raw output
        st.subheader("📄 Raw Output")
        st.code(responce.content, language="json")

        # try parsing
        try:
            parsed = parser.parse(responce.content)

            st.subheader("📊 Structured Output")

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Title:**", parsed.title)
                st.write("**Author:**", parsed.author)
                st.write("**Year:**", parsed.release_year)
                st.write("**Rating:**", parsed.rating)

            with col2:
                st.write("**Genres:**", ", ".join(parsed.genre))
                st.write("**Main Characters:**", ", ".join(parsed.main_character))

            st.subheader("📝 Summary")
            st.write(parsed.summary)

        except Exception as e:
            st.error("Parsing failed (model output not perfectly structured)")