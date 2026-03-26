from langchain_community.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()


# Prompt
prompt = PromptTemplate(
    template="Write a summary for the following poem - \n {poem}?",
    input_variables=["poem"]
)

# ✅ Initialize model properly
model = GoogleGenerativeAI(
    model="gemini-2.5-flash",   # or "gemini-1.5-pro"
    temperature=0.7
)

parser = StrOutputParser

loader = TextLoader("D:\\Langchain\\Hugging_face.text", encoding="utf-8")



docs = loader.load()

print(type(docs))

print(len(docs))

print(docs[0])

chain = prompt | model | parser

print(chain.invoke({'poem': docs[0].page_content}))