# AQA_KEYCLOAK
AQA (Asymetric Question Answering) inference microservices with Keycloak


# Run API Services and Chat UI
1. Preparation
    - `.env` file for reading system config
    - add documents to folder `docs` and mount volume
    - add prompt template to folder `templates` and mount volume
2. Clone the repository and run with docker-compose:
    ```bash
    git clone https://github.com/trungthanhnguyenn/aqa_keycloak.git
    cd aqa_keycloak
    docker compose up -d
    ```
    - After the docker run, go to keycloak 'http://localhost:8080/realms/fastapi-realm'
    - Choose **fastapi-realm** in "Manage realms"
    - Create user for **fastapi-realm** after that fill your realm and user info in **llm_gradio.py**
    - The API service will be available at `http://localhost:2222`.
    - The Gradio UI Chat will be available at `http://localhost:2223`.

3. To stop the service:
    ```bash
    docker compose down
    ```

# Structure Repository
    ```
    aqa_keycloakkeycloak/
    ├── conf.d/                 # Configuration files
    ├── docs/                   # Document storage for RAG
    ├── models/                 # LLM models storage
    ├── qdrant_hub/             # Qdrant configuration
    ├── scripts/                # Bash scripts runnning
    ├── src/                    # Source code
    │   ├── api/                # API endpoints
    │   ├── db/                 # Database interfaces
    │   ├── module/             # Core wrapper modules
    │   ├── services/           # Service Logic implementations
    │   └── utils/              # Utility functions
    ├── templates/              # Template storage for Prompt
    ├── docker-compose.yml      # Docker compose config
    ├── Dockerfile.client       # Client container setup
    ├── llm_gradio.py           # Gradio UI to chat
    ├── mainapi.py              # Root FastAPI
    ├── realm-export-1.json     # Keycloak realm export
    └── .env                    # Environment variables
    ```

## License
[AGPL v3.0](LICENSE).<br>
Copyright @ 2025 [Pythera](https://github.com/pytheralab/sati). All rights reserved.
