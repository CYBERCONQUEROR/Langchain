import streamlit as st
import os
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA

# Page Configuration
st.set_page_config(
    page_title="YouTube RAG Intelligence",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stTextInput > div > div > input {
        background-color: #262730;
        color: #ffffff;
    }
    .stChatMessage {
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    llm_model = st.selectbox("Select LLM Model", ["gemma4:latest", "llama3:latest", "mistral:latest"], index=0)
    embed_model = st.selectbox("Select Embedding Model", ["qwen3-embedding:4b", "nomic-embed-text:latest"], index=0)
    k_value = st.slider("Retrieval K-Value", 1, 10, 4)
    st.divider()
    st.markdown("🌐 **Language Settings**")
    lang_codes = st.text_input("Transcription Codes (comma-separated)", value="en, hi, es, fr, de, it, ja, ko")
    st.divider()
    if st.button("🗑️ Clear Session"):
        st.session_state.clear()
        st.rerun()

st.title("📽️ YouTube RAG Intelligence")
st.markdown("---")

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "current_video" not in st.session_state:
    st.session_state.current_video = None

# Video URL Input
video_url = st.text_input("🔗 Paste YouTube Video URL:", placeholder="https://www.youtube.com/watch?v=...")

# Processing Logic
def process_video(url, embed_m):
    try:
        with st.status("🛠️ Processing Video...", expanded=True) as status:
            st.write("📥 Fetching Transcript...")
            # Parse language codes
            langs = [l.strip() for l in lang_codes.split(",")]
            loader = YoutubeLoader.from_youtube_url(url, add_video_info=False, language=langs)
            docs = loader.load()
            
            st.write("✂️ Chunking Content...")
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(docs)
            
            st.write("🧠 Generating Embeddings & Indexing...")
            embeddings = OllamaEmbeddings(model=embed_m)
            vector_store = FAISS.from_documents(chunks, embeddings)
            
            status.update(label="✅ Indexing Complete!", state="complete", expanded=False)
            return vector_store
    except Exception as e:
        st.error(f"Error: {e}")
        return None

if video_url and video_url != st.session_state.current_video:
    st.session_state.vector_store = process_video(video_url, embed_model)
    st.session_state.current_video = video_url
    st.session_state.messages = [] # Reset chat for new video

# Chat Interface
if st.session_state.vector_store:
    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    if prompt := st.chat_input("Ask a question about the video:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Setup QA Chain
                llm = OllamaLLM(model=llm_model, temperature=0)
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=st.session_state.vector_store.as_retriever(search_kwargs={"k": k_value}),
                    return_source_documents=True
                )
                
                response = qa_chain.invoke({"query": prompt})
                answer = response["result"]
                sources = response["source_documents"]
                
                st.markdown(answer)
                
                with st.expander("📚 View Sources"):
                    for i, doc in enumerate(sources):
                        st.info(f"Source {i+1}:\n{doc.page_content[:300]}...")

        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("👋 Enter a YouTube URL in the field above to start chatting with its content locally.")
