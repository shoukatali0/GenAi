
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

from pydantic import BaseModel, Field
from typing import Optional, List
from langchain_core.output_parsers import PydanticOutputParser 

load_dotenv()

from langchain_mistralai import ChatMistralAI

model = ChatMistralAI(model = "mistral-small-2603")

class BookInfo(BaseModel):
    
    title: str = Field(description="The full title of the book")
    author: str = Field(description="The name of the author")
    release_year: Optional[int] = Field(description="The year the book was published")
    genre: List[str] = Field(description="List of genres")
    main_character: List[str] = Field(description="List of main characters")
    rating: Optional[float] = Field(description="The book's rating out of 10")
    summary: str = Field(description="A brief summary of the content")


parser = PydanticOutputParser(pydantic_object=BookInfo)

prompt = ChatPromptTemplate.from_messages([

("system", """

You are an expert literary analyst and intelligent information extractor.
{follow_instructions}
"""),
("human", "{input_text}"),
]
)

para = input("give the data : ")

final_prompt = prompt.invoke({"input_text": para, 
"follow_instructions": parser.get_format_instructions()})

responce = model.invoke(final_prompt)

print(responce.content)