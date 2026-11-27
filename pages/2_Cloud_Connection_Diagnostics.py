
import streamlit as st
from config.clients_streamlit import llm, embeddings

st.title("Cloud API Connection Diagnostics")

# LLM test
st.subheader("1. Qwen LLM Connection")

if st.button("Test LLM Connection"):
    try:
        response = llm.invoke(
            "Reply with exactly: LLM connection successful"
        )
        st.success("LLM connection successful")
        st.write("Response:", response.content)

    except Exception as exc:
        st.error("LLM connection failed")
        st.write("Error type:", type(exc).__name__)
        st.write("Error:", str(exc))

# Embedding test
st.subheader("2. BGE Embedding Connection")

if st.button("Test Embedding Connection"):
    try:
        vector = embeddings.embed_query(
            "Employee annual leave policy"
        )
        st.success("Embedding connection successful")
        st.write("Vector dimensions:", len(vector))

    except Exception as exc:
        st.error("Embedding connection failed")
        st.write("Error type:", type(exc).__name__)
        st.write("Error:", str(exc))