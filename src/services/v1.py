from typing import List, Dict, Any
from src.module.module import BaseModule, LLMModule
from src.services.interface import InterfaceService
from src.db.interface import InterfaceDatabase
from src.utils.utils import read_file

import hashlib
import json
import os
import numpy as np
import glob 
 

class ServicesV1(InterfaceService):
    """Manager for Services"""
    def __init__(
        self,
        query_module: BaseModule,
        context_module: BaseModule, 
        rerank_module: BaseModule,
        chunker_module: BaseModule,
        llm_module: LLMModule,
        chunk_db: InterfaceDatabase,
    ):
        super().__init__()
        self.query_module = query_module
        self.context_module = context_module
        self.rerank_module = rerank_module
        self.chunker_module = chunker_module
        self.llm_module = llm_module
        self.chunk_db = chunk_db

    def retrieve_chunks(self, query: str, chunker_id, **kwargs) -> List[Dict[str, Any]]:
        # cast List to query here
        query_embed = self.query_module.embed(
            [query],
            padding=True, 
            truncation=True, 
            return_tensors="np"
        )
        # search for chunks
        retrieve_chunks = self.chunk_db.search(
            chunk_emb=query_embed[0], # List[List[float]] -> List[float] 
            chunker_id=chunker_id,
            top_k=kwargs.get('top_k', 5)
        )
        return retrieve_chunks

    def insert_chunks(self, chunks: List[dict], chunker_id: str) -> dict:
        """
        chunks = [{
            "text": ...,
            "doc_id": ...,
        }]
        """
        if not chunks:
            return {"Error": "No chunks to insert!"}
        
        # format the chunks
        texts = [chunk['text'] for chunk in chunks]
        ids = [hashlib.md5(text.encode()).hexdigest() for text in texts]
        embeds = self.context_module.embed(
            texts,
            padding=True, 
            truncation=True, 
            return_tensors="np"
        )
        insert_dicts = {
            'ids': ids,
            'payloads': [chunk for chunk in chunks],
            'vectors': embeds,
        }
        try:
            self.chunk_db.insert(insert_dicts, chunker_id)
        except Exception as e:
            return {"Error": f"Error inserting chunks: {e}"}
        return {"Success": "Chunks inserted!"}
    
    def merge_subtexts_fix(self, list_sub_text, max_tokens=None):
        merged_texts = []  
        current_num_token = 0
        current_merge = ""  
        # 
        if max_tokens is None:
            max_tokens = self.context_module.tokenizer.model_max_length
        #
        for subtext in list_sub_text:
            # 
            text_responses = self.context_module.tokenizer(subtext)
            num_chars = np.shape(text_responses['input_ids'])[-1]
            #
            if current_num_token + num_chars > max_tokens:
                if current_merge:  
                    merged_texts.append(current_merge.strip())
                current_num_token = num_chars
                current_merge = subtext 
            else:
                current_num_token += num_chars
                current_merge += " " + subtext 
        #
        if current_merge:
            merged_texts.append(current_merge.strip())
        return merged_texts
    
    def rerank(self, query: str, chunks: List[dict], **kwargs) -> List[Dict[str, Any]]:
        """
        Rerank the chunks
        """
        if not chunks:
            return []
        # rerank
        scores = []
        for i in range(0, len(chunks)):
            score = self.rerank_module.rerank(
                query, 
                chunks[i]['payload']['text'],
                return_tensors="pt"
            )
            scores.append(score)
        # sort by score
        sorted_chunks = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        return [chunk for chunk, score in sorted_chunks]

    def chunking(self, text: str) -> List:
        # tokenizer text
       return self.chunker_module.chunking(text)
    