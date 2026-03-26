from langchain_community.document_loaders import CSVLoader

loader = ''

docs =loader.load ()

print(docs[0].page_content)