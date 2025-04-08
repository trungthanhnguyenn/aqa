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
        with open("cache.txt", "w") as f:
            f.write("")
    else:
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
        chunks = requests.post(
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
        if chunks.get("error"):
            print("Error inserting chunks:", chunks.get("error"))
            continue
    return "Insert docs successfully"

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
    # instruction = f"`{inst}`"
    # 
    context_template = ""
    for i, c in enumerate(chunks):
        context_template += f"{c}\n\n"
    # 
    # prompt_template = prompt_template.format(
    #     instruction=instruction,
    #     query=query,
    #     context=context_template,
    # )
    
    prompt_template = open("templates/prompt.txt", "r").read()
    prompt_template = jinja2.Template(prompt_template).render(
        context=context_template,
        prompt=query,
    )

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

    # generate text
    generate_kwargs = dict(
        # input_ids=input_ids,
        max_tokens=max_length, 
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=rp,    
    )
    payload = {
        "prompt": conversation,
        "sampling_parameters": generate_kwargs
    }
    
    llm_url = f"{API_URL}/generate_stream"
    response = requests.post(url=llm_url, json=payload, stream=True)

    # yield conversation[len(prompt):]
    # Print the outputs.
    end_think = True
    generated_text = ""
    for output in response.iter_lines():
        text_output = output.decode("utf-8")
        # for token in text_output.split(" "):
        #     yield token

        if text_output == "</think>":
            end_think = True

        # if end_think and text_output != "</think>": deepseek
        if end_think and text_output != "<|start_header_id|>assistant<|end_header_id|>": # llama
            generated_text += text_output + "\n"

            # print("generated_text", generated_text)
            print("text_output", text_output)
            yield generated_text
            time.sleep(0.3)


    # return generated_text

    # return generated_text

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
            # source_lang = gr.Textbox(
            #     label="Source Lang(Auto-Detect)",
            #     value="English",
            # )
            # target_lang = gr.Dropdown(
            #     label="Target Lang",
            #     value="Spanish",
            #     choices=LANG_LIST,
            # )
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

            
            # inst = gr.Textbox(
            #     label="Instruction",
            #     value="Answering these following question.",
            #     lines=3,
            # )
            # prompt = gr.Textbox(
            #     label="Prompt",
            #     value="""""",
            #     lines=8,
            # )
                
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
    # gr.Markdown(LICENSE)
    
    # source_text.change(lang_detector, source_text, source_lang)
    # source_text.change(lang_detector, source_text)
    # submit.click(fn=translate, inputs=[source_text, source_lang, target_lang, inst, prompt, max_length, temperature, top_p, rp], outputs=[output_text])
    
    docs_path = "docs/*.txt"
    status = insert_docs(docs_path)
    print(status)

    # submit.click(fn=run, inputs=[source_text, inst, prompt, max_length, temperature, top_p, rp], outputs=[output_text])
    submit.click(fn=run, inputs=[source_text, max_length, temperature, top_p, rp], outputs=[output_text])
    #
    upload_button.click(fn=handle_file_upload, inputs=[file_upload], outputs=[output_text])

    # with gr.Accordion("Tutorials", open=True):
    #     gr.Markdown("## Instruction")
    #     gr.Textbox(value="For examples: Supposed that you are the professor in education.",
    #             label="The instruction that describes a command.")

    #     gr.Markdown("\n\n\n### Prompt: Format template used to achieve tasks.")
    #     gr.Textbox(value="Answer questions based on a given context",
    #                 label="1. Question Answering")

    #     gr.Markdown("\n\n\n### Question: The input text that you want to ask.")
    #     gr.Markdown("### Generation Configuration:")
    #     gr.Textbox(value="The maximum length of the generated text.", label="1. Max Length")
    #     gr.Textbox(value="Lower values make the output more deterministic, while higher values increase randomness.", label='2. Temperature')
    #     gr.Textbox(value="Lower values make the model consider fewer options, while higher values allow more diverse outputs.", label="3. Top_p")
    #     gr.Textbox(value="Lower values make the model consider repeated tokens less likely to be chosen again", label="4. Repetition Penalty")
        

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")