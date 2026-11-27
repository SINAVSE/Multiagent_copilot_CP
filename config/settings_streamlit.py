import base64
import ssl
from types import SimpleNamespace

import streamlit as st


def get_secret(name, default=None):
    value = st.secrets.get(name, default)
    if value is None or value == "":
        if default is None:
            raise ValueError(f"Missing Streamlit Secret: {name}")
    return value


def create_ssl_context(secret_name):
    encoded = get_secret(secret_name)

    try:
        cert_bytes = base64.b64decode(encoded, validate=True)
        cert_text = cert_bytes.decode("utf-8-sig").strip()

        if "-----BEGIN CERTIFICATE-----" not in cert_text:
            raise ValueError("PEM certificate header not found")

        return ssl.create_default_context(cadata=cert_text)

    except Exception as exc:
        raise ValueError(
            f"Could not load certificate from Streamlit Secret: {secret_name}"
        ) from exc


settings = SimpleNamespace(
    api_key=get_secret("API_KEY"),
    llm_base_url=get_secret("LLM_BASE_URL"),
    llm_model=get_secret("LLM_MODEL", "Qwen/Qwen3-8B"),
    llm_ssl_context=create_ssl_context("LLM_SSL_CERT_B64"),

    embeddings_key=get_secret("EMBEDDINGS_KEY"),
    embedding_base_url=get_secret("EMBEDDING_BASE_URL"),
    embedding_model=get_secret(
        "EMBEDDING_MODEL",
        "BAAI/bge-small-en-v1.5",
    ),
    embedding_ssl_context=create_ssl_context("EMBEDDING_SSL_CERT_B64"),
)