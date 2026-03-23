from google.generativeai import list_models
import google.generativeai as genai

genai.configure(api_key="AIzaSyC7PKnpCVYDTIhccmvlLjmpGsts3io5G14")

for m in list_models():
    print(m.name)