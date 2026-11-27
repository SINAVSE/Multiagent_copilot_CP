
import tempfile
from pathlib import Path
from types import SimpleNamespace

import streamlit as st


def get_secret(name, default=None):
    value = st.secrets.get(name, default)
    if value is None or value == "":
        if default is None:
            raise ValueError(f"Missing Streamlit Secret: {name}")
    return value


def create_certificate(secret_name):
    content = get_secret(secret_name)

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".crt",
        encoding="utf-8",
        delete=False,
    ) as cert_file:
        cert_file.write(str(content).strip() + "\n")
        return Path(cert_file.name)


settings = SimpleNamespace(
    api_key=get_secret("API_KEY"),
    llm_base_url=get_secret("LLM_BASE_URL"),
    llm_model=get_secret("LLM_MODEL", "Qwen/Qwen3-8B"),
    llm_ssl_cert_path=create_certificate("LLM_SSL_CERT"),

    embeddings_key=get_secret("EMBEDDINGS_KEY"),
    embedding_base_url=get_secret("EMBEDDING_BASE_URL"),
    embedding_model=get_secret(
        "EMBEDDING_MODEL",
        "BAAI/bge-small-en-v1.5",
    ),
    embedding_ssl_cert_path=create_certificate("EMBEDDING_SSL_CERT"),
)