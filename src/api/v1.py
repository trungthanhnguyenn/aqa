from typing import List
from pydantic import BaseModel

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from src.configs.settings import Settings
from src.services.v1 import ServicesV1
from src.module.module import EmbeddingModule, RerankModule, ChunkerModule, LLMModule
from src.db.qdrant_db import QdrantChunksDB

####################
# Init modules
####################

# Read environment variables
settings = Settings()
grpc = settings.protocol.lower() == "grpc"

# Retrieval Configs
query_module = EmbeddingModule(
    model_path=settings.rtv_model_serving_path,
    model_name=settings.query_model_name,
    model_version=settings.query_model_version,
    model_server_url=settings.rtv_triton_url,
    is_grpc=grpc,
    config_file_name=settings.query_config_file_name,
)
context_module = EmbeddingModule(
    model_path=settings.rtv_model_serving_path,
    model_name=settings.ctx_model_name,
    model_version=settings.ctx_model_version,
    model_server_url=settings.rtv_triton_url,
    is_grpc=grpc,
    config_file_name=settings.ctx_config_file_name,
)
rerank_module = RerankModule(
    model_path=settings.rtv_model_serving_path,
    model_name=settings.rerank_model_name,
    model_version=settings.rerank_model_version,
    model_server_url=settings.rtv_triton_url,
    is_grpc=grpc,
    config_file_name=settings.rerank_config_file_name,
)

# Chunker Configs
chunker_module = ChunkerModule(
    model_path=settings.sati_model_serving_path,
    model_name=settings.chunker_model_name,
    model_version=settings.chunker_model_version,
    model_server_url=settings.sati_triton_url,
    is_grpc=grpc,
    config_file_name=settings.chunker_config_file_name,
)

# LLM Configs
llm_module = LLMModule(
    model_path=settings.llm_model_serving_path,
    model_name=settings.model_llm_name,
    model_server_url=settings.llm_triton_url,
    tokenizer_name=settings.llm_tokenizer_name,
    streaming=settings.streaming_response,
    config_file_name=settings.llm_config_file_name,
)

# Database Configs
db = QdrantChunksDB(url=settings.qdrant_db)

# Sevices V1 
services = ServicesV1(
    query_module=query_module,
    context_module=context_module,
    rerank_module=rerank_module,
    chunker_module=chunker_module,
    llm_module=llm_module,
    chunk_db=db
)

####################
# API Router
####################

router = APIRouter(prefix="/v1")

####################
# Retrieval API
@router.get("/chunks")
async def get_chunks() -> JSONResponse:
    # add remote with async func
    chunker_id = settings.qdrant_collection_name
    response = services.chunk_db.get_chunks(chunker_id=chunker_id)
    return JSONResponse(content=response)

@router.post("/chunks")
async def insert_chunks(chunks: List[dict]) -> JSONResponse:
    # add remote with async func
    chunker_id = settings.qdrant_collection_name
    response = services.insert_chunks(chunks, chunker_id)
    return JSONResponse(content=response)

@router.delete("/chunks/{chunk_id}")
async def delete_chunk_id(chunk_id: str) -> JSONResponse:        
    # add remote with async func
    chunker_id = settings.qdrant_collection_name
    response = services.chunk_db.delete(chunk_ids=[chunk_id], chunker_id=chunker_id, doc_id=None)
    return JSONResponse(content=response)

####################
# Chunker API
class Item(BaseModel):
    text: str
    metadata: dict = None

@router.get("/documents/{doc_id}")
async def get_doc_id(doc_id: str) -> JSONResponse:
    chunker_id = settings.qdrant_collection_name
    # add remote with async func
    response = services.chunk_db.get_chunks_by_doc_id(doc_id=doc_id, chunker_id=chunker_id)
    if not response:
        return JSONResponse(content={"Error": "No chunks of document found!"})
    # format response
    return JSONResponse(content=response)

@router.delete("/documents/{doc_id}")
async def delete_doc_id(doc_id: str) -> JSONResponse:
    chunker_id = settings.qdrant_collection_name
    # add remote with async func
    response = services.chunk_db.delete(doc_id=doc_id, chunker_id=chunker_id, chunk_ids=None)
    return JSONResponse(content=response)

# send n docs to the server
@router.post("/documents")
async def chunker(request: Item) -> JSONResponse:
    # chunking
    chunks = services.chunking(request.text)
    max_chars = services.context_module.tokenizer.model_max_length # 512
    merge_chunks = services.merge_subtexts_fix(
        chunks, 
        max_tokens=max_chars
    )
    # format to db
    format_insert_chunks = [
        {
            "filename": request.metadata.get("filename", "unknown"),
            "file_size": request.metadata.get("file_size", 0),
            "created_time": request.metadata.get("created_time", 0),
            "text": sub_text,
        }
        for sub_text in merge_chunks 
    ]
    # insert to db
    try:
        for i in range(0, len(format_insert_chunks), settings.ctx_batch_size):
            batch_text_format = format_insert_chunks[i:i+settings.ctx_batch_size]
            # insert chunks to db
            response = await services.insert_chunks(batch_text_format)
            if response.status_code != 200:
                print(response.json())
                return JSONResponse(content={"error": "Insert chunks failed!"}, status_code=500)
        
        return JSONResponse(content={
            "success": f"Insert number of {len(format_insert_chunks)} to {settings.qdrant_collection_name} DB successfully!"
        })
    
    except Exception as e:
        return JSONResponse(content={"error": "Insert chunks failed!"})

####################
# QA API
@router.post("/queries")
async def retrieve_chunks(query: str, is_rerank: bool = False) -> JSONResponse:
    chunker_id = settings.qdrant_collection_name
    # add remote with async func
    chunks = services.retrieve_chunks(query, chunker_id)
    if not chunks:
        return JSONResponse(content={"Error": "No chunks found!"})
    
    # rerank
    if is_rerank:   
        chunks = services.rerank(
            query=query,
            chunks=chunks,
        )
    return JSONResponse(content=chunks)


####################
# LLM API
@router.post("/chat_template")
async def chat_template(text: str) -> JSONResponse:
    """
    """
    # call tokenizer
    chat_template = [{"role": "user", "content": text}]
    chat_template = services.llm_module.tokenizer.apply_chat_template(chat_template, tokenize=False)
    return JSONResponse(content=chat_template)


@router.post("/generate_stream")
async def generate_stream(request: Request):
    """
    """
    user_input = await request.json()
    prompt_text = user_input.get("prompt", "")
    sampling_parameters = user_input.get("sampling_parameters", {})
    sampling_parameters["stream"] = True

    if not prompt_text:
        return {"error": "Missing 'prompt' in request payload"}

    payload = {
        "text_input": prompt_text,
        "parameters": sampling_parameters,
    }
    return StreamingResponse(
        services.llm_module.generate_stream(payload=payload), 
        media_type="application/json"
    )

@router.post("/generate")
async def generate(request: Request) -> JSONResponse:
    """
    """
    user_input = await request.json()
    prompt_text = user_input.get("prompt", "")
    sampling_parameters = user_input.get("sampling_parameters", {})

    if not prompt_text:
        return {"error": "Missing 'prompt' in request payload"}

    payload = {
        "text_input": prompt_text,
        "parameters": sampling_parameters,
    }
    return JSONResponse(content=services.llm_module.generate(payload=payload))

