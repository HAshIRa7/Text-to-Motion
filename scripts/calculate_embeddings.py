import asyncio, uuid, os
from functools import partial
import numpy as np
from tqdm.asyncio import tqdm_asyncio
from vllm.config import PoolerConfig
from vllm import AsyncEngineArgs, AsyncLLMEngine, PoolingParams, SamplingParams
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm
import tyro
from text_to_motion import collect_data
from vllm.sampling_params import StructuredOutputsParams
from transformers import AutoTokenizer
import json
from typing import Dict

SENTINEL = object()

SCHEMA = {
    "type": "object",
    "properties": {
        "natural": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 3},
    },
    "required": ["natural"],
}

def save_npz(filepath: str, answ: Dict):
    with np.load(filepath, allow_pickle=True) as data:
        dct = dict(data)
    for i, text in enumerate(answ["natural"]):
        dct[f'text_{i}'] = text
    np.savez(filepath, **dct)

def load(motions_dir: str, motion_file: str):
    filepath = os.path.join(motions_dir, motion_file)
    with np.load(filepath, allow_pickle=True) as data:
        try:
            text = str(data['text'])
        except:
            text = ''
    return (filepath, text)

async def embed_worker(model, tokenizer, sampling_params, sem, item, queue):
    fp, text = item
    
    messages = [
        {"role": "system", "content":
            "You paraphrase motion captions for a robotics dataset. "
            "Preserve the meaning exactly. Never rename motion skill terms and also preserve time. Produce: 3 natural rephrasings."},
        {"role": "user", "content":
            f"Caption: {text}\n"},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)

    async with sem:
        rid = str(uuid.uuid4())
        res = None
        async for out in model.generate(
            prompt,
            sampling_params,
            request_id=rid,
        ):
            res = out
        try:
            answ = json.loads(res.outputs[0].text)
            await queue.put((fp, answ))
        except Exception as e:
            print(f'exception {e} on {fp}')

async def save_consumer(queue, loop, pbar):
    while True:
        item = await queue.get()
        if item is SENTINEL:
            queue.task_done()
            return
        fp, answ = item
        try:
            await loop.run_in_executor(None, save_npz, fp, answ)
        except Exception as e:
            print(f"save failed for {fp}: {e}")
        finally:
            pbar.update(1)
            queue.task_done()

async def async_main(motions_dir: str,
                     embed_concurrency: int = 128,
                     save_workers: int = 32):
    files = sorted(os.listdir(motions_dir))
    items: list[tuple[str, str]] = []
    
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(load, motions_dir, motion_file): motion_file for motion_file in files}
        
        with tqdm(total=len(futures), desc="Processing") as pbar:
            for future in as_completed(futures):
                items.append(future.result())
                pbar.update(1) 

    model_name = "Qwen/Qwen3-4B-Instruct-2507"
    engine_args = AsyncEngineArgs(
        model=model_name,
        gpu_memory_utilization=0.95,
        data_parallel_size=8,
    )
    
    model = AsyncLLMEngine.from_engine_args(engine_args)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    sampling_params = SamplingParams(max_tokens=2048, structured_outputs=StructuredOutputsParams(json=SCHEMA))
    loop = asyncio.get_running_loop()
    sem = asyncio.Semaphore(embed_concurrency)
    queue = asyncio.Queue(maxsize=embed_concurrency * 2)

    with tqdm(total=len(items), desc="Embedding+Saving") as pbar:
        consumers = [
            asyncio.create_task(save_consumer(queue, loop, pbar))
            for _ in range(save_workers)
        ]
        producers = [
            asyncio.create_task(embed_worker(model, tokenizer, sampling_params, sem, item, queue))
            for item in items
        ]
        await asyncio.gather(*producers)
        for _ in range(save_workers):
            await queue.put(SENTINEL)
        await asyncio.gather(*consumers)
        
def calculate_embeddings(
    motions_dir: str = 'motions',
    new_motions_dir: str = 'postprocessed_motions',
    motions_len_min: int = 51,
    motions_len_max: int = 2500,
    embed_concurrency: int = 224,
):
    # collect_data(motions_dir, new_motions_dir, motions_len_min, motions_len_max)
    asyncio.run(async_main(new_motions_dir, embed_concurrency))


if __name__ == '__main__':
    tyro.cli(calculate_embeddings)