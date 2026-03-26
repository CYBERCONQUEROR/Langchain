from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("Audio Dataset (WAV files).pdf")

docs = loader.load()

print(len(docs))

print(docs[0].page_content)
