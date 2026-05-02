from vllm import LLM
from typing import List
import torch

class EmbedderModel:
    
    def __init__(self, model_name = "Qwen/Qwen3-Embedding-4B", task='embed', max_gpu_utilization=0.2):
        
        self.embedder_model = LLM(
            model=model_name, 
            convert=task, 
            gpu_memory_utilization=max_gpu_utilization,
        ) 
        
        self.null_token_embedding = self.embedder_model.embed([''])[0].outputs.embedding
        self.embed_dim = self.null_token_embedding.shape[-1]
    
        
    def encode_text_batch(self, text_batch: List[str], replacement_probs = 0.15, inference=False) -> torch.Tensor:
        batch_len = len(text_batch)
        if not inference:
            mask = torch.rand(batch_len) > replacement_probs
        else:
            mask = torch.ones(batch_len)
        outputs = self.embedder_model.embed(text_batch)
        embeddings = torch.tensor([o.outputs.embedding for o in outputs])
        
        return torch.where(mask, embeddings, self.null_token_embedding.unsqueeze(dim=0))