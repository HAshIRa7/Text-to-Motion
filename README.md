# Text-to-Motion

**Controllable motion generation for humanoid robots.**

Text-to-Motion generates physically-plausible reference trajectories for the
[Unitree G1](https://www.unitree.com/g1) humanoid (29 actuated DoF) directly from
natural-language prompts. Given a sentence such as *"a person walks forward then
stands on one leg"*, the model produces a full motion clip — joint positions,
root orientation and root trajectory — that can be replayed in a viewer or used
as **synthetic reference data** for training whole-body controllers (WBC) or
vision-language-action (VLA) policies.

Generation is based on **flow matching** with **classifier-free guidance**, and
the network is implemented as a custom efficient transformer with Triton / FlashAttention
kernels for fast, memory-light training and inference on a single A100.

| | |
|---|---|
| Robot | Unitree G1 (29 DoF) |
| Generative paradigm | Flow matching (rectified-flow / linear interpolation) |
| Text encoder | FLAN-T5-XL encoder (frozen, a T5 variant) |
| Conditioning | Cross-attention to T5 token embeddings + CFG |
| Control rate | 50 FPS (`dt = 0.02 s`) |
| Recommended hardware | NVIDIA A100-SXM4-80GB |

---

## Table of contents

- [Architecture](#architecture)
- [Data representation](#data-representation)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Data preparation](#data-preparation)
- [Training](#training)
  - [Single node](#single-node)
  - [Multi-node (distributed)](#multi-node-distributed)
- [Inference and visualization](#inference-and-visualization)
- [Notes](#notes)

---

## Architecture

The model is a conditional **flow-matching** network, `FlowMatchingNet`
(`src/text_to_motion/diffusion_net.py`), built from two parts:

1. **A frozen text encoder** — `google/flan-t5-xl` loaded as a `T5EncoderModel`
   in `bfloat16`. It is a T5-family encoder whose last hidden states provide the
   conditioning token embeddings. Its weights are frozen during training
   (`requires_grad_(False)`).

2. **A trainable flow transformer** — `EfficientTransformer`
   (`src/efficient_model/transformer.py`), which regresses the flow velocity
   field over the motion sequence.

### Flow transformer (`EfficientTransformer`)

```
motion frames x_t  ──▶ in_linear (64 → hidden)
                       + gated sinusoidal positional encoding
                       │
            ┌──────────▼───────────┐   ×  num_layers
            │  TransformerBlock     │
            │   RMSNorm             │
            │   self-attention (RoPE, FlashAttention varlen)
            │   RMSNorm             │
            │   cross-attention → T5 text embeddings (FlashAttention varlen)
            │   RMSNorm             │
            │   SwiGLU feed-forward │
            └──────────┬───────────┘
                       │  FusedAdaLN time-step modulation (after every block)
                       ▼
                    out_linear (hidden → 64)  ──▶ predicted velocity field
```

Each `TransformerBlock` applies, with residual connections:

- **Self-attention** over the motion frames, with **RoPE** rotary embeddings and
  variable-length packed `flash_attn_varlen_func` (no padding; sequences are
  concatenated and indexed by `cu_seqlens`).
- **Cross-attention** from the motion frames (queries) to the T5 text token
  embeddings (keys/values), again using variable-length FlashAttention.
- A **SwiGLU** feed-forward block (gpt-oss style: `down((up + 1) · silu(gate))`,
  with clamping and `alpha = 1.702`).

Time conditioning is injected by a shared **`FusedAdaLNModulation`** layer
(`src/efficient_model/adaln.py`) applied after every transformer block: the
scalar flow time `t` is turned into a sinusoidal embedding and mapped to
`(gamma, beta, alpha)` that modulate the activations (adaptive layer norm).

### Custom kernels / efficiency

- **Zero-centered RMSNorm** (`norm.py`) — `y = x / rms(x) · (1 + weight)`,
  `weight` initialized to zero, with a `torch.compile`-d forward/backward.
- **Fused SwiGLU** (`swiglu.py`) — Triton forward/backward kernels plus a
  memory-efficient autograd `Function` that recomputes activations in the
  backward pass to save memory (based on the Liger-Kernel SwiGLU).
- **FlashAttention** variable-length kernels (`attention.py`) for both self- and
  cross-attention, so batches are processed as packed sequences without padding.
- **RoPE** rotary positional embeddings on self-attention.

### Default configuration & parameter count

The architecture hyperparameters live in `src/text_to_motion/config.py`
(`TransformerConfig`):

| Field | Default | Meaning |
|---|---|---|
| `input_dim` / `output_dim` | 64 | per-frame motion vector size |
| `hidden_dim` | 1024 | transformer model width |
| `embed_dim` | 2048 | T5 text-embedding width (cross-attn K/V input) |
| `intermediate_dim` | 2048 | SwiGLU inner width |
| `num_layers` | 10 | transformer blocks |
| `num_heads` | 8 | attention heads |
| `max_seq_len` | 4096 | max motion length (frames) |
| `rope_theta` | 10000.0 | RoPE base |

With these defaults the **trainable flow transformer is ≈176M parameters**
(the frozen FLAN-T5-XL encoder is additional and not optimized). The parameter
count scales with `hidden_dim`, `intermediate_dim` and `num_layers` — the project
targets a flow transformer in the **~235M-parameter** class, which is reached by
scaling these values. The text encoder remains a frozen T5 variant in all
configurations.

### Training objective

Flow matching with linear interpolation (`scripts/train.py`):

```
x0 ~ N(0, I)                      # noise
x1 = data (a real motion clip)    # target
t  ~ U(0, 1)                      # per-sample flow time
x_t = t · x1 + (1 - t) · x0       # interpolated point
loss = MSE( model(x_t, text, t),  x1 - x0 )   # predict the velocity field
```

Classifier-free guidance is enabled by randomly dropping the text prompt during
training (≈15% of samples get an empty string in `make_collate_fn`). At inference
the conditional and unconditional velocity fields are combined with a
`guidance_scale` (default 3.0).

---

## Data representation

Each motion frame is a **64-dimensional** vector (all values standardized using
dataset statistics stored in `statistic_collector.pkl`):

| Slice | Dim | Quantity |
|---|---|---|
| `0:29` | 29 | joint positions |
| `29:30` | 1 | root roll |
| `30:31` | 1 | root pitch |
| `31:33` | 2 | planar linear velocity (yaw-aligned x/y) |
| `33:34` | 1 | yaw angular velocity |
| `34:63` | 29 | joint velocities |
| `63:64` | 1 | root height |

At inference, `postprocess_motion` (`scripts/viser_generate_play.py`) inverts the
standardization and integrates velocities back into a root trajectory
(`convert_lin_vel_xy_to_root_pos`) and root quaternion
(`convert_roll_pitch_ang_vel_to_quat`) using the quaternion / Euler helpers in
`src/text_to_motion/math.py`. Output is a dict with `joint_names`, `joint_pos`,
`body_pos_w` (root position) and `body_quat_w` (root quaternion, scalar-first).

---

## Repository layout

```
src/
  text_to_motion/
    config.py            # TransformerConfig dataclass
    diffusion_net.py     # FlowMatchingNet: T5 encoder + flow transformer, CFG sampling steps
    temporal_dataset.py  # HumanoidDataset + variable-length collate_fn (CFG text dropout)
    utils.py             # raw-motion preprocessing, dataset statistics, quaternion integration
    math.py              # quaternion / Euler / yaw math (numpy)
  efficient_model/
    transformer.py       # EfficientTransformer + TransformerBlock
    attention.py         # RoPE + FlashAttention varlen self/cross attention
    swiglu.py            # Triton fused + memory-efficient SwiGLU MLP
    norm.py              # zero-centered RMSNorm
    adaln.py             # FusedAdaLN time-step modulation
    positional_encoding.py # gated sinusoidal positional encoding
scripts/
  calculate_embeddings.py # preprocess raw motions + compute text embeddings (vLLM / BGE-M3)
  cluster_data.py         # MiniBatchKMeans clustering of motion text embeddings
  train.py                # DDP flow-matching training loop
  viser_generate_play.py  # text -> motion generation + live Viser visualization
  viser_motion_play.py    # replay a motion .npz file in Viser
motions/                  # example raw motion clips (.npz)
statistic_collector.pkl   # precomputed dataset normalization statistics
pyproject.toml            # uv project / dependencies
```

---

## Installation

The project is managed with [`uv`](https://github.com/astral-sh/uv) and targets
**Python 3.12** with **CUDA 12.8** wheels (PyTorch, FlashAttention 2.8.3,
liger-kernel, Triton). An NVIDIA A100-SXM4-80GB is the reference GPU.

```bash
# 1. install uv (Linux / macOS)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. clone and install
git clone https://github.com/HAshIRa7/Text-to-Motion.git
cd Text-to-Motion
uv sync
source .venv/bin/activate
```

`uv sync` pulls a prebuilt FlashAttention wheel and the CUDA-12.8 PyTorch index
defined in `pyproject.toml`, so a working CUDA 12.8 driver stack is required.

---

## Data preparation

The training set is built from raw G1 motion captures (`.npz` files containing
`joint_pos`, `joint_vel`, `body_pos_w`, `body_quat_w`, `body_lin_vel_w`,
`body_ang_vel_w` and per-segment text `metadata`). Two preprocessing utilities
are provided:

**1. Preprocess motions + compute statistics + text embeddings.** This slices
raw clips into training windows, derives the 64-d motion vectors, writes them to
`postprocessed_motions/`, (re)computes `statistic_collector.pkl`, and attaches a
text embedding to each clip using a [vLLM](https://github.com/vllm-project/vllm)
`BAAI/bge-m3` engine:

```bash
python scripts/calculate_embeddings.py \
  --motions-dir motions \
  --new-motions-dir postprocessed_motions \
  --motions-len-min 51 \
  --motions-len-max 2500
```

**2. (Optional) Cluster the prompts** for analysis / balancing with MiniBatch
K-Means over the BGE-M3 embeddings; produces `cluster_model.pkl`:

```bash
python scripts/cluster_data.py --motions-dir postprocessed_motions --n-clusters 5
```

> The text **conditioning at train time uses the FLAN-T5-XL encoder** (tokenized
> on the fly in the collate function). The BGE-M3 embeddings produced above are an
> auxiliary signal used for data analysis / clustering, not for conditioning.

---

## Training

Training (`scripts/train.py`) is written for **PyTorch DistributedDataParallel
(DDP)** and reads its rank/world topology from `torchrun` environment variables
(`LOCAL_RANK`, `RANK`, `WORLD_SIZE`). The process group uses the **NCCL** backend
and the data is sharded with a `DistributedSampler`. Because the script always
calls `dist.init_process_group(...)`, it should be launched with `torchrun`
even on a single GPU.

Key defaults: `batch_size = 16` per process, `AdamW` at `lr = 1e-4` with an
`ExponentialLR(gamma=0.9)` per-epoch decay, mixed precision (`bfloat16` autocast +
`GradScaler`), 1000 epochs. The frozen T5 encoder runs in `bfloat16`; checkpoints
are written to `checkpoints/model_new_weight_{epoch}.pth` and TensorBoard logs to
`logs/`. `statistic_collector.pkl` and the `postprocessed_motions/` folder must
exist before launching.

### Single node

One GPU:

```bash
torchrun --standalone --nproc_per_node=1 scripts/train.py
```

All GPUs on one machine (e.g. 8×A100):

```bash
torchrun --standalone --nproc_per_node=8 scripts/train.py
```

`--standalone` makes `torchrun` pick a free rendezvous endpoint automatically, so
no master address/port is needed for a single node.

Monitor training:

```bash
tensorboard --logdir logs
```

### Multi-node (distributed)

Launch `torchrun` once per node. Pick one node as the rendezvous master and give
every node the same `--master_addr` / `--master_port`, a unique `--node_rank`,
and the total `--nnodes`. Example for **2 nodes × 8 GPUs = 16 processes**:

```bash
# ---- node 0 (rendezvous master, e.g. 10.0.0.1) ----
torchrun \
  --nnodes=2 \
  --node_rank=0 \
  --nproc_per_node=8 \
  --master_addr=10.0.0.1 \
  --master_port=29500 \
  scripts/train.py

# ---- node 1 ----
torchrun \
  --nnodes=2 \
  --node_rank=1 \
  --nproc_per_node=8 \
  --master_addr=10.0.0.1 \
  --master_port=29500 \
  scripts/train.py
```

The effective global batch size is `batch_size × nnodes × nproc_per_node`
(here `16 × 2 × 8 = 256`). Make sure every node has identical code,
`statistic_collector.pkl`, and access to the same `postprocessed_motions/`
dataset (e.g. on shared storage), and that the NCCL port is open between nodes.

A modern rendezvous form using a shared endpoint also works:

```bash
torchrun --nnodes=2 --node_rank=$NODE_RANK --nproc_per_node=8 \
  --rdzv_backend=c10d --rdzv_endpoint=10.0.0.1:29500 \
  scripts/train.py
```

---

## Inference and visualization

**Generate a motion from text and watch it live.** `scripts/viser_generate_play.py`
loads a checkpoint, spins up a [Viser](https://viser.studio/main/) server, loads
the G1 URDF, and lets you type a prompt, set the clip length and guidance scale
with sliders, and play the generated motion on the robot. Sampling uses an
EDM-style schedule with Euler steps and classifier-free guidance:

```bash
python scripts/viser_generate_play.py --checkpoint-path checkpoints/model_new_weight_1.pth
```

Then open the printed Viser URL, enter a prompt (default *"A person walks
forward"*), and click **Generate Motion**. The `time` slider controls clip length
in seconds (frames = `50 × seconds`) and `guidance scale` controls CFG strength.

**Replay an existing motion file** (raw or generated `.npz`) without the model:

```bash
python scripts/viser_motion_play.py --motion-file <name>.npz --motion-folder motions
```

All scripts use [`tyro`](https://github.com/brentyi/tyro), so every function
argument is exposed as a `--flag` on the command line.

---

## Notes

- **Hardware.** Training and inference are tuned for an NVIDIA A100-SXM4-80GB.
  FlashAttention varlen kernels, Triton SwiGLU and the frozen FLAN-T5-XL encoder
  assume a recent CUDA stack (CUDA 12.8 wheels are pinned in `pyproject.toml`).
- **Robot.** The G1 URDF is fetched via `robot_descriptions` (`g1_description`);
  joint ordering is remapped to the URDF actuated-joint order at visualization
  time (`match_joint_names`).
- **Determinism.** Attention kernels are called with `deterministic=True`.

## License / contact

Author: Egor (`egormipt@yandex.ru`). See repository for license details.
