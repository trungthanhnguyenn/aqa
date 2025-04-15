from typing import List

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.services.v1 import ServicesV1
from src.module.module import EmbeddingModule, RerankModule, ChunkerModule, LLMModule
from src.db.qdrant_db import QdrantChunksDB


from pydantic import BaseModel


class Settings(BaseSettings):
    #
    query_model_name: str = None
    query_model_version: int = None
    query_batch_size: int = None
    ctx_model_name: str = None
    ctx_model_version: int = None
    ctx_batch_size: int = None
    rerank_model_name: str = None
    rerank_model_version: int = None
    rerank_batch_size: int = None
    rtv_triton_url: str = "localhost:8000"
    rtv_model_serving_path: str = "/models"
    #
    chunker_model_name: str = None
    chunker_model_version: int = None
    chunker_batch_size: int = None
    sati_triton_url: str = "localhost:8000"
    sati_model_serving_path: str = "/models"
    # 
    model_llm_name: str = None
    llm_tokenizer_name: str = None
    streaming_response: bool = False
    llm_triton_url: str = "localhost:8000"
    llm_model_serving_path: str = "/models"
    # 
    protocol: str = "HTTP"
    verbose: bool = False
    async_set: bool = False
    qdrant_db: str = None
    qdrant_collection_name: str = "retrieval"
    top_k: int = 5
    threshold: float = 0.5

    model_config = SettingsConfigDict(env_file=".env", extra='ignore')

# Parse environment variables
settings = Settings()
grpc = settings.protocol.lower() == "grpc"

####################
# Init modules
####################

# rtv
query_module = EmbeddingModule(
    model_path=settings.rtv_model_serving_path,
    model_name=settings.query_model_name,
    model_version=settings.query_model_version,
    model_server_url=settings.rtv_triton_url,
    is_grpc=grpc
)
context_module = EmbeddingModule(
    model_path=settings.rtv_model_serving_path,
    model_name=settings.ctx_model_name,
    model_version=settings.ctx_model_version,
    model_server_url=settings.rtv_triton_url,
    is_grpc=grpc
)
rerank_module = RerankModule(
    model_path=settings.rtv_model_serving_path,
    model_name=settings.rerank_model_name,
    model_version=settings.rerank_model_version,
    model_server_url=settings.rtv_triton_url,
    is_grpc=grpc
)

# chunker
chunker_module = ChunkerModule(
    model_path=settings.sati_model_serving_path,
    model_name=settings.chunker_model_name,
    model_version=settings.chunker_model_version,
    model_server_url=settings.sati_triton_url,
    is_grpc=grpc
)

# llm
llm_module = LLMModule(
    model_path=settings.llm_model_serving_path,
    model_name=settings.model_llm_name,
    model_server_url=settings.llm_triton_url,
    tokenizer_name=settings.llm_tokenizer_name,
    streaming=settings.streaming_response
)

# db
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

app = FastAPI(prefix="/v1")

origins = [
    "http://localhost"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello Bigger Applications!"}

@app.get("/chunks")
async def get_chunks() -> JSONResponse:
    # add remote with async func
    chunker_id = settings.qdrant_collection_name
    response = services.chunk_db.get_chunks(chunker_id=chunker_id)
    return JSONResponse(content=response)

@app.post("/chunks")
async def insert_chunks(chunks: List[dict]) -> JSONResponse:
    # add remote with async func
    chunker_id = settings.qdrant_collection_name
    response = services.insert_chunks(chunks, chunker_id)
    return JSONResponse(content=response)

@app.delete("/chunks/{chunk_id}")
async def delete_chunk_id(chunk_id: str) -> JSONResponse:        
    # add remote with async func
    chunker_id = settings.qdrant_collection_name
    response = services.chunk_db.delete(chunk_ids=[chunk_id], chunker_id=chunker_id, doc_id=None)
    return JSONResponse(content=response)

@app.get("/documents/{doc_id}")
async def get_doc_id(doc_id: str) -> JSONResponse:
    chunker_id = settings.qdrant_collection_name
    # add remote with async func
    response = services.chunk_db.get_chunks_by_doc_id(doc_id=doc_id, chunker_id=chunker_id)
    if not response:
        return JSONResponse(content={"Error": "No chunks of document found!"})
    # format response
    return JSONResponse(content=response)

@app.delete("/documents/{doc_id}")
async def delete_doc_id(doc_id: str) -> JSONResponse:
    chunker_id = settings.qdrant_collection_name
    # add remote with async func
    response = services.chunk_db.delete(doc_id=doc_id, chunker_id=chunker_id, chunk_ids=None)
    return JSONResponse(content=response)

@app.post("/queries")
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


class Item(BaseModel):
    text: str
    metadata: dict = None

# send n docs to the server
@app.post("/documents")
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
        for i in range(0, len(format_insert_chunks), 4):
            batch_text_format = format_insert_chunks[i:i+4]
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


@app.post("/generate_stream")
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

@app.post("/generate")
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


@app.post("/chat_template")
async def chat_template(text: str) -> JSONResponse:
    """
    """
    # call tokenizer
    chat_template = [{"role": "user", "content": text}]
    chat_template = services.llm_module.tokenizer.apply_chat_template(chat_template, tokenize=False)
    return JSONResponse(content=chat_template)