# AQA
AQA (Asymetric Question Answering) inference microservices


# Run API Services and Chat UI
1. Prepare `.env` to pull images and down models 
2. Clone the repository and run with docker-compose:
    ```bash
    git clone https://github.com/pytheralab/aqa.git
    cd aqa
    docker-compose up -d
    ```
    - The API service will be available at `http://localhost:2222`.
    - The Gradio UI Chat will be available at `http://localhost:2223`.

3. To stop the service:
    ```bash
    docker-compose down
    ```


## License
[AGPL v3.0](LICENSE).<br>
Copyright @ 2025 [Pythera](https://github.com/pytheralab/sati). All rights reserved.
