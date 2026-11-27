
import httpx
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config.settings_streamlit import settings


# Create HTTP clients using the SSL contexts from Streamlit Secrets
llm_http_client = httpx.Client(
    verify=settings.llm_ssl_context,
    timeout=60.0,
)

embedding_http_client = httpx.Client(
    verify=settings.embedding_ssl_context,
    timeout=60.0,
)


# LLM client
llm = ChatOpenAI(
    model=settings.llm_model,
    api_key=settings.api_key,
    base_url=settings.llm_base_url,
    http_client=llm_http_client,
    temperature=0,
)


# LLM client for self-consistency
llm_self_consistency = ChatOpenAI(
    model=settings.llm_model,
    api_key=settings.api_key,
    base_url=settings.llm_base_url,
    http_client=llm_http_client,
    temperature=0.7,
)


# Embedding client
embeddings = OpenAIEmbeddings(
    model=settings.embedding_model,
    api_key=settings.embeddings_key,
    base_url=settings.embedding_base_url,
    http_client=embedding_http_client,
    tiktoken_enabled=False,
    embedding_ctx_length=512,
)