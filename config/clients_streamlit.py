
import httpx
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config.settings_streamlit import settings


llm_http_client = httpx.Client(
    verify=str(settings.llm_ssl_cert_path),
    timeout=60.0,
)

embedding_http_client = httpx.Client(
    verify=str(settings.embedding_ssl_cert_path),
    timeout=60.0,
)

llm = ChatOpenAI(
    model=settings.llm_model,
    api_key=settings.api_key,
    base_url=settings.llm_base_url,
    http_client=llm_http_client,
    temperature=0,
)

llm_self_consistency = ChatOpenAI(
    model=settings.llm_model,
    api_key=settings.api_key,
    base_url=settings.llm_base_url,
    http_client=llm_http_client,
    temperature=0.7,
)

embeddings = OpenAIEmbeddings(
    model=settings.embedding_model,
    api_key=settings.embeddings_key,
    base_url=settings.embedding_base_url,
    http_client=embedding_http_client,
    tiktoken_enabled=False,
    embedding_ctx_length=512,
)