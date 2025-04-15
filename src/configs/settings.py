from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    #
    rtv_triton_url: str = "localhost:8000"
    rtv_model_serving_path: str = "/models"
    sati_triton_url: str = "localhost:8000"
    sati_model_serving_path: str = "/models"
    llm_triton_url: str = "localhost:8000"
    llm_model_serving_path: str = "/models"
    #
    query_config_file_name: str = None
    query_model_name: str = None
    query_model_version: int = None
    query_batch_size: int = None
    #
    ctx_config_name_file: str = None
    ctx_model_name: str = None
    ctx_model_version: int = None
    ctx_batch_size: int = None
    #
    rerank_config_file_name: str = None
    rerank_model_name: str = None
    rerank_model_version: int = None
    rerank_batch_size: int = None
    #
    chunker_config_file_name: str = None
    chunker_model_name: str = None
    chunker_model_version: int = None
    chunker_batch_size: int = None
    # 
    llm_config_file_name: str = None
    model_llm_name: str = None
    llm_tokenizer_name: str = None
    streaming_response: bool = False
    # 
    protocol: str = "HTTP"
    verbose: bool = False
    async_set: bool = False
    qdrant_db: str = None
    qdrant_collection_name: str = "retrieval"
    top_k: int = 5
    threshold: float = 0.5
    #
    model_config = SettingsConfigDict(env_file=".env", extra='ignore')
