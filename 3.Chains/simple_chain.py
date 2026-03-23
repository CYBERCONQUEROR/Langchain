from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Prompt
prompt = PromptTemplate(
    template="What is the capital of {country}?",
    input_variables=["country"]
)

# ✅ Initialize model properly
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",   # or "gemini-1.5-pro"
    temperature=0.7
)

# Output parser
parser = StrOutputParser()

# Chain
chain = prompt | model | parser

# Run
result = chain.invoke({"country": "France"})
print(result)

chain.get_graph()