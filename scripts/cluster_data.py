import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import os
from sklearn.cluster import MiniBatchKMeans
import joblib
import tyro
import asyncio
import random

SENTINEL = object()

def load_embed(filepath: str):
    with np.load(filepath, allow_pickle=True) as data:
        dct = dict(data)
    emb = dct['emb'][-1]
    emb = emb / np.linalg.norm(emb)
    return emb

async def load_worker(sem, fp, loop, queue):
    async with sem:
        try:
            emb = await loop.run_in_executor(None, load_embed, fp)
            await queue.put(emb)
        except Exception as e:
            print(f"save failed for {fp}: {e}")

async def fit_kmeans(queue, loop, pbar, model, storage_embs, batch_size):
    while True:
        emb = await queue.get()
        if emb is SENTINEL:
            queue.task_done()
            if len(storage_embs) > 0:
                await loop.run_in_executor(None, model.partial_fit, np.stack(storage_embs, axis=0))
            pbar.update(len(storage_embs))
            storage_embs.clear()
            return
        storage_embs.append(emb)
        if len(storage_embs) == batch_size:
            await loop.run_in_executor(None, model.partial_fit, np.stack(storage_embs, axis=0))
            storage_embs.clear()
            pbar.update(batch_size)


async def async_main(motions_dir: str, concurrency_size: int, batch_size: int, n_clusters: int):
    files = sorted(os.listdir(motions_dir))
    random.shuffle(files)
    model = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size)
    loop = asyncio.get_running_loop()
    sem = asyncio.Semaphore(concurrency_size)
    queue = asyncio.Queue(maxsize=concurrency_size * 2)
    storage_embs = []
    
    with tqdm(total=len(files), desc='Cluster Model Fitting') as pbar:
        consumers = [
            asyncio.create_task(fit_kmeans(queue, loop, pbar, model, storage_embs, batch_size))
        ]
        producers = [
            asyncio.create_task(load_worker(sem, os.path.join(motions_dir, file), loop, queue))
            for file in files
        ]
        
        await asyncio.gather(*producers)
        await queue.put(SENTINEL)
        await asyncio.gather(*consumers)
        
        joblib.dump(model, "cluster_model.pkl")
        
    
def cluster_data(
    motions_dir: str = 'postprocessed_motions',
    concurrency_size: int = 128,
    batch_size: int = 1024, 
    n_clusters: int = 5,
):
    asyncio.run(async_main(motions_dir, concurrency_size, batch_size, n_clusters))

if __name__ == '__main__':
    tyro.cli(cluster_data)