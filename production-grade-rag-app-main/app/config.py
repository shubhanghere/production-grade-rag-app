from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    anthropic_api_key: str
    voyage_api_key: str
    llm_model_name: str = "claude-sonnet-4-6"
    # A different model from llm_model_name so eval judging is independent of the
    # model under test (the system does not grade its own homework).
    eval_judge_model: str = "claude-opus-4-8"
    embedding_model: str = "voyage-3"
    chroma_persist_dir: str = "./chroma_db"
    # Final number of chunks passed to the LLM after fusion.
    retrieval_k: int = 5
    # Candidates each retriever (dense, BM25) returns before RRF fusion.
    retrieval_pool: int = 20
    # Voyage cross-encoder used to rerank the fused candidate pool.
    rerank_model: str = "rerank-2.5"
    # Voyage rate limits (requests/min, tokens/min). With a payment method on the
    # account, these are the higher "standard tier" limits; set conservatively
    # below the real ceiling. The 429 backoff in _embed is the safety net.
    # (The no-payment reduced tier was 3 RPM / 10K TPM.)
    voyage_rpm: int = 60
    voyage_tpm: int = 1_000_000
    # Reject single uploads larger than this (chars). A safety cap until Stage 6
    # adds real splitting of large unstructured documents.
    max_upload_chars: int = 50_000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()
