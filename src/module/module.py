from typing import List, Dict, Any, Tuple
from trism import TritonModel
from transformers import AutoTokenizer
import numpy as np
import json
import httpx
import os

class BaseModule:
    """Factory class"""
    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    def __init__(self, model_path: str, model_name: str, model_version: int, model_server_url: str, is_grpc: bool = False, **kwargs):
        # 
        model_name_or_path = os.path.join(model_path, model_name, str(model_version))
        # 
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, **kwargs,
        )
        #
        self._model = TritonModel(
            model=model_name,                 # Model name.
            version=model_version,            # Model version.
            url=model_server_url,              # Triton Server URL.
            grpc=is_grpc                      # Use gRPC or Http.
        )
        #
        self._config = self.read_config_model(model_name_or_path, **kwargs)

    def read_config_model(self, model_name_or_path: str, **kwargs):
        """
        Get the config of the model
        """
        config_file_name = kwargs.get('config_file_name', None)
        config_path = os.path.join(model_name_or_path, config_file_name)
        if not os.path.exists(config_path):
            return {"error": f"Config file {config_file_name} not found!"}
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            return config
        except json.JSONDecodeError:
            return {"error": f"Config file {config_file_name} is not a valid JSON!"}

class EmbeddingModule(BaseModule):

    def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        text_responses = self._tokenizer(
            texts, 
            **kwargs
        )
        # padding=True, 
        # truncation=True, 
        # return_tensors="np"
        try:
            outputs: Dict[Any] = self._model.run(data = [
                text_responses['input_ids'], 
                text_responses['attention_mask'], 
                text_responses['token_type_ids']
            ])
            outputs = list(outputs.values())[0] # BxLx768
            outputs = outputs.reshape(len(texts), -1, 768)[:, 0].tolist() # Bx768
            return outputs
        except Exception as e:
            print(f"Error embedding text: {e}")
            return []
        

class RerankModule(BaseModule):
    # Encode text
    def rerank(self, query: str, context: str, **kwargs) -> List[float]:

        # Tokenize sentences
        encoded_pair = self._tokenizer(
            query,
            context,
            **kwargs,  # return_tensors="pt"
        )
        for key in encoded_pair:
            encoded_pair[key] = encoded_pair[key].numpy()
        # Compute token embeddings
        score = self._model.run(
            data=[
                encoded_pair['input_ids'],
                encoded_pair['attention_mask'],
                encoded_pair['token_type_ids']
            ]
        )['logits']
        # Get the score
        score = score.reshape(-1).tolist()
        return score

class ChunkerModule(BaseModule):

    def chunking(self, text: str) -> List:
        
        texts_token = self._tokenizer(
            text,
            return_offsets_mapping=True,
            verbose=False,
            add_special_tokens=False,
            padding=False,
            truncation=False
        )

        texts_responses = []
        # all_output =[]
        times = 0
        start = 0
        input_model = []
        while start + 512 < len(texts_token['input_ids']):
            input_model = []
            for inp in self._model.inputs:
                input_model.append(np.array([texts_token[inp.name][start:start + 512]]))
            outputs = self._model.run(data=input_model)[self._model.outputs[0].name][0] # -> 1 x bz x length -> bz x length
            outputs =[value_outputs[0] for value_outputs in outputs]
            # create batch
            all_index = [index + 1 for index, value in enumerate(outputs) if value > 0]
            all_index = [512*times + index for index in all_index]
            if len(all_index) > 0:
                texts_responses.extend(all_index)
                start = texts_responses[-1]
            else:
                start += 512
            times += 1
        # final batch
        for inp in self._model.inputs:
            input_model.append(np.array([texts_token[inp.name][start:]]))
        outputs = self._model.run(data=input_model)[self._model.outputs[0].name][0]
        outputs =[value_outputs[0] for value_outputs in outputs]
        all_index = [index + 1 for index, value in enumerate(outputs) if value > 0]
        all_index = [start + index for index in all_index]
        texts_responses.extend(all_index)
        # decode
        start = 0
        return_value = []
        if texts_responses == []:
            return_value.append(self._tokenizer.decode(texts_token['input_ids'][start:]))
            return return_value
        
        for index in texts_responses:
            return_value.append(self._tokenizer.decode(texts_token['input_ids'][start:index]))
            start = index
        return_value = [value for value in return_value if value != ""]
        return return_value



class LLMModule(BaseModule):

    @property
    def config(self):
        return self._config

    # override the new TritonModel
    def __init__(self, model_path: str, model_name: str, model_version: int, model_server_url: str, is_grpc: bool = False, **kwargs):
        # 
        model_name_or_path = os.path.join(model_path, model_name, str(model_version))
        # 
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, **kwargs,
        )
        #! FIX
        self._model = TritonLLM(
            model=model_name,                 # Model name.
            version=model_version,            # Model version.
            url=model_server_url,              # Triton Server URL.
            grpc=is_grpc                      # Use gRPC or Http.
        )
        #
        self._config = self.read_config_model(model_name_or_path, **kwargs)


    #! FIX
    async def fetch_vllm_stream(self, payload):
        """
        Gửi request đến Triton và xử lý response dạng streaming từng token.
        - ✅ Đẩy từng token lên API ngay lập tức.
        - ✅ Kiểm tra xem Triton có gửi token trùng hay không.
        - ✅ Xóa khoảng trắng dư thừa giữa các từ.
        """
        global previous_token
        previous_token = None 

        async with httpx.AsyncClient(timeout=120.0) as client:
            model_url = f"http://{self.model_url}/v2/models/{self.model_name}/generate_stream"
            async with client.stream("POST", model_url, json=payload) as response:
                if response.status_code != 200:
                    error_msg = json.dumps({
                        "error": f"Server returned {response.status_code}",
                        "detail": (await response.aread()).decode("utf-8")
                    })
                    print(error_msg)
                    # yield f"{error_msg}\n\n"
                    yield ""
                    return

                async for line in response.aiter_lines():
                    if line:
                        try:
                            clean_line = line.replace("data: ", "", 1)
                            data = json.loads(clean_line)
                            text_output = data.get("text_output", "")
                            yield text_output
                                
                        except json.JSONDecodeError:
                            error_msg = json.dumps({"error": "JSON parse error", "raw": clean_line})
                            # yield f"{error_msg}\n\n"
                            print(error_msg)
                            yield ""


