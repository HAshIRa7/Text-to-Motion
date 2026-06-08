import asyncio, uuid, os
from functools import partial
import numpy as np
from tqdm.asyncio import tqdm_asyncio
from vllm.config import PoolerConfig
from vllm import AsyncEngineArgs, AsyncLLMEngine, PoolingParams
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm
import tyro
from text_to_motion import collect_data
SENTINEL = object()

def save_npz(filepath: str, embedding: np.ndarray):
    with np.load(filepath, allow_pickle=True) as data:
        dct = dict(data)
    dct['emb'] = embedding
    np.savez(filepath, **dct)

def load(motions_dir: str, motion_file: str):
    filepath = os.path.join(motions_dir, motion_file)
    with np.load(filepath, allow_pickle=True) as data:
        try:
            text = str(data['text'])
        except:
            text = ''
    return (filepath, text)

async def embed_worker(model, sem, item, queue):
    fp, text = item
    async with sem:
        rid = str(uuid.uuid4())
        res = None
        async for out in model.encode(
            text,
            PoolingParams(),
            request_id=rid,
        ):
            res = out
        emb = res.outputs.data.float().cpu().numpy()
    await queue.put((fp, emb))

async def save_consumer(queue, loop, pbar):
    while True:
        item = await queue.get()
        if item is SENTINEL:
            queue.task_done()
            return
        fp, emb = item
        try:
            await loop.run_in_executor(None, save_npz, fp, emb)
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

    engine_args = AsyncEngineArgs(
        model="BAAI/bge-m3",
        convert="embed",
        gpu_memory_utilization=0.95,
        runner="pooling",
        max_model_len=8192,
        pooler_config=PoolerConfig(
            task="token_embed",
            # seq_pooling_type="ALL",
            use_activation=False, # in question !!!!!!
        )
    )
    model = AsyncLLMEngine.from_engine_args(engine_args)
    loop = asyncio.get_running_loop()
    sem = asyncio.Semaphore(embed_concurrency)
    queue = asyncio.Queue(maxsize=embed_concurrency * 2)

    with tqdm(total=len(items), desc="Embedding+Saving") as pbar:
        consumers = [
            asyncio.create_task(save_consumer(queue, loop, pbar))
            for _ in range(save_workers)
        ]
        producers = [
            asyncio.create_task(embed_worker(model, sem, item, queue))
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
    embed_concurrency: int = 32,
):
    collect_data(motions_dir, new_motions_dir, motions_len_min, motions_len_max)
    asyncio.run(async_main(new_motions_dir, embed_concurrency))


if __name__ == '__main__':
    tyro.cli(calculate_embeddings)