import gradio as gr
# import spaces
import os
import numpy as np
from dotenv import load_dotenv
import requests
import glob
import hashlib
from typing import List
import time
import jinja2
import logging

from src.utils.utils import read_file

load_dotenv()

# API
API_URL = os.environ.get("API_URL", "http://localhost:1997")
GOOGLE_URL_API = os.environ.get("GOOGLE_URL_API", "")

#

##########
# Function
##########

    
def handle_file_upload(files):
    """
    Handles the file upload process:
    - Saves the uploaded file locally.
    - Calls the `insert_docs` function to process and upload the file to the database.
    """
    print("file", files)
    if files is not None:
        for file in files:
            # Save the uploaded file locally
            file_path = os.path.join("uploaded_files", file.name)
            os.makedirs("uploaded_files", exist_ok=True)
            with open(file_path, "wb") as f:
                f.write(f.read())

            # Process and upload the file to the database
            status = insert_docs(file_path)
            return status
    return "No file uploaded."
    
    
def insert_docs(docs_path: str):
    # read doc
    file_paths = glob.glob(docs_path)
    cache_doc_file = "cache.txt"
    if not os.path.exists(cache_doc_file):
        # create file
        logging.info("Cache file not found, creating a new one.")
        with open("cache.txt", "w") as f:
            f.write("")
    else:
        logging.info("Cache file found, reading existing hashes.")
        with open("cache.txt", "r") as f:
            cache_doc_file = f.read().split("\n")

    # chunking doc
    for file in file_paths:
        doc = read_file(file)
        filename = file.split("/")[-1]
        hash_filename = hashlib.md5(filename.encode()).hexdigest()
        # not insert the same file
        if hash_filename in cache_doc_file:
            continue
        else:
            with open("cache.txt", "a") as f:
                f.write(hash_filename + "\n")
        # other metadata   
        file_size = os.path.getsize(file)
        file_stat = os.stat(file)
        created_time = file_stat.st_ctime  # Creation time
        # chunking
        chunks_status = requests.post(
            f"{API_URL}/documents", 
            json={
                "text": doc, 
                "metadata": {
                    "filename": filename,
                    "file_size": file_size,
                    "created_time": created_time,
                }
            }
        ).json()
        if chunks_status.get("error"):
            print("Error inserting chunks:", chunks_status.get("error"))
            continue
    return "Process completed!"

def retrieve_docs(query: str, top_k: int = 5):
    response = requests.post(
        url=f"{API_URL}/queries",
        json={
            "query": query,
            "is_rerank": False,
        },
    )
    retrieve_docs = response.json()
    return retrieve_docs



def parse_prompt_template(instruction: str, query: str, chunks: List[str], prompt_template: str):
    #
    context_template = ""
    for i, c in enumerate(chunks):
        context_template += f"{c}\n\n"
    # 
    prompt_template = open("templates/prompt.txt", "r").read()
    prompt_template = jinja2.Template(prompt_template).render(
        context=context_template,
        prompt=query,
    )
    # apply chat template
    prompt = requests.post(
        url=f"{API_URL}/chat_template",
        params={
            'text': prompt_template
        },
    ).json()
    #
    print("prompt_template", prompt)
    return prompt

    

def run(
    source_text: str, 
    # inst: str, 
    # prompt: str, 
    max_length: int,
    temperature: float,
    top_p: float,
    rp: float
    ): 
    print(f'Question is - {source_text}')
    
    # chunks
    retrieve_chunks = requests.post(
        url=f"{API_URL}/queries",
        params={
            'query': source_text,
            "is_rerank": False
        },
    ).json()
    print("retrieve_chunks", retrieve_chunks)
    retrieve_chunks = [doc['payload']["text"] for doc in retrieve_chunks]
    
    # format template prompt
    conversation = parse_prompt_template(
        # instruction=inst, 
        instruction=None,
        query=source_text, 
        # prompt_template=prompt,
        prompt_template=None,
        chunks=retrieve_chunks
    )

    # generate response
    generate_kwargs = dict(
        # input_ids=input_ids,
        max_tokens=max_length, 
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=rp,    
    )
    payload = {
        "prompt": conversation,
        "sampling_parameters": generate_kwargs,
        "show_thinking": True,
    }
    # get response
    llm_url = f"{API_URL}/generate_stream"
    response = requests.post(url=llm_url, json=payload, stream=True)

    # stream the outputs.
    generated_text = ""
    for text_output in response.iter_lines(decode_unicode=True):
        generated_text += text_output + "\n"
        print("text_output", text_output)
        yield generated_text


CSS = """
    h1 {
        text-align: center;
        display: block;
        height: 10vh;
        align-content: center;
    }
    footer {
        visibility: hidden;
    }
"""

chatbot = gr.Chatbot(height=600)

with gr.Blocks(theme="soft", css=CSS) as demo:
    # gr.Markdown(TITLE)
    with gr.Row():
        with gr.Column(scale=1):

            max_length = gr.Slider(
                label="Max Length",
                minimum=512,
                maximum=8192,
                value=4096,
                step=8,
            )
            temperature = gr.Slider(
                label="Temperature",
                minimum=0,
                maximum=1,
                value=0.3,
                step=0.1,
            )
            top_p = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                step=0.1,
                value=1.0,
                label="top_p",
            )
            rp = gr.Slider(
                minimum=0.0,
                maximum=2.0,
                step=0.1,
                value=1.2,
                label="Repetition penalty",
            )
            file_upload = gr.File(
                label="Upload Files", 
                file_types=[".pdf", ".docx", ".txt"],
                file_count="multiple"
            )

                
        with gr.Column(scale=4):
            source_text = gr.Textbox(
                label="Question",
                value="Quỹ BEQ là gì?",
                lines=10,
            )
            output_text = gr.Textbox(
                label="Response",
                lines=10,
                show_copy_button=True,
            )
    with gr.Row():
        upload_button = gr.Button(value="Upload")
        submit = gr.Button(value="Submit")
        
        clear = gr.ClearButton([source_text, output_text])

    #* Insert documents
    docs_path = "/docs/*.txt"
    status = insert_docs(docs_path)
    print(status)
    #
    submit.click(fn=run, inputs=[source_text, max_length, temperature, top_p, rp], outputs=[output_text])
    #
    upload_button.click(fn=handle_file_upload, inputs=[file_upload], outputs=[output_text])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")