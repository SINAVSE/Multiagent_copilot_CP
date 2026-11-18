import httpx
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from config.settings import settings

# Separate HTTP clients with SSL certificate verification
llm_http_client = httpx.Client(
    verify=str(settings.llm_ssl_cert_path),
    timeout=60.0,
)

embedding_http_client = httpx.Client(
    verify=str(settings.embedding_ssl_cert_path),
    timeout=60.0,
)

# LLM client: Consistency
llm = ChatOpenAI(
    model=settings.llm_model,
    api_key=settings.api_key,
    base_url=settings.llm_base_url,
    http_client=httpx.Client(
        verify=str(settings.llm_ssl_cert_path)
    ),
    temperature=0,
)
#With randomness
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
    http_client=httpx.Client(
        verify=str(settings.embedding_ssl_cert_path)
    ),
)