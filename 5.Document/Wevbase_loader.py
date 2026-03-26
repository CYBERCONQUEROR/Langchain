from langchain_community.document_loaders import WebBaseLoader
from pydantic_core import Url
url ='https://blinkit.com/s/?q=speaker%20for%20laptop'

loader = WebBaseLoader(url)

docs = loader.load()

print(docs[0].page_content)