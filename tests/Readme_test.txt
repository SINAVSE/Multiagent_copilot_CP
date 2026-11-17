# Configuration and Client Verification Commands

## 1. Verify settings

```powershell
python -c "from config.settings import settings; print('LLM Model:', settings.llm_model); print('Embedding Model:', settings.embedding_model); print('LLM certificate exists:', settings.llm_ssl_cert_path.is_file()); print('Embedding certificate exists:', settings.embedding_ssl_cert_path.is_file())"
```

## 2. Verify client initialization

```powershell
python -c "from config.clients import llm, embeddings; print('LLM client initialized:', llm is not None); print('Embedding client initialized:', embeddings is not None)"
```

## 3. Test LLM connectivity

```powershell
python -m pytest tests/test_llm_connection.py -v -s
```

## 4. Test embedding connectivity

```powershell
python -m pytest tests/test_embedding_connection.py -v -s
```

## 5. Run all tests

```powershell
python -m pytest tests/ -v
```