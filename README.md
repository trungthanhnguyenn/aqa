# AQA
AQA (Asymetric Question Answering) inference microservices


# Run API Services and Chat UI
1. Preparation
    - `.env` file for reading system config
    - add documents to folder `docs` and mount volume
    - add prompt template to folder `templates` and mount volume
2. Clone the repository and run with docker-compose:
    ```bash
    git clone https://github.com/pytheralab/aqa.git
    cd aqa
    docker compose up -d
    ```
    - The API service will be available at `http://localhost:2222`.
    - The Gradio UI Chat will be available at `http://localhost:2223`.

3. To stop the service:
    ```bash
    docker compose down
    ```

# Structure Repository
    ```
    aqa/
    ├── conf.d/                 # Configuration files
    │   └── huggingface.json    # HuggingFace model configs
    ├── docs/                   # Document storage for RAG
    ├── models/                 # LLM models storage
    ├── scripts/                # Bash scripts runnning
    ├── src/                    # Source code
    │   ├── api/                # API endpoints
    │   ├── db/                 # Database interfaces
    │   ├── module/             # Core wrapper modules
    │   ├── services/           # Service Logic implementations
    │   └── utils/              # Utility functions
    ├── templates/              # Template storage for Prompt
    ├── docker-compose.yml      # Docker compose config
    ├── llm_gradio.py           # Gradio UI to chat
    ├── mainapi.py              # Root FastAPI 
    └── .env                    # Environment variables
    ```

## License
[AGPL v3.0](LICENSE).<br>
Copyright @ 2025 [Pythera](https://github.com/pytheralab/sati). All rights reserved.
