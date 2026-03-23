from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(temperature=0)
response = llm("What is the capital of France?")
print(response)

