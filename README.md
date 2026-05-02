# Text-to-Motion project

This is project for generating reference motions on G1 Unitree humanoid robot using text prompts. Can be used for synthetic data generation for training WBC or VLA.
In this repository you find training scripts, architecture and inference code with vizualization using [`Viser`](https://viser.studio/main/). Solution build on flow-matching generation with 31M parameters decoder like transformer with Triton kernel optimizations for efficient training and inference. Requiremets: NVIDIA A100-SXM4-80GB

<table>
  <tr>
    <td align="center">
      <video width="950" controls loop muted playsinline>
        <source src="images/walk_run_dance.mp4" type="video/mp4">
      </video>
      <p align="center"><strong>📝 Walk, Run, Dance</strong></p>
    </td>
    <td align="center">
      <video width="950" controls loop muted playsinline>
        <source src="images/stay_on_one_leg.mp4" type="video/mp4">
      </video>
      <p align="center"><strong>📝 Stay on one leg</strong></p>
    </td>
  </tr>
</table>

## Quick start

### 1. Install `uv`
[`uv`](https://github.com/astral-sh/uv) is an extremely fast Python package manager and resolver written in Rust. Install it using your preferred method:

**Linux / macOS**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone project, install project

```bash
git clone https://github.com/HAshIRa7/Text-to-Motion.git
uv sync
source .venv/bin/activate
``` 

### 3. Launch Training 
This command launch flow matching diffusion training, requires ±20 GB GPU memory.
```bash
python scripts/train.py
```

### 4. Generate Reference
To infer your net you need to specify in code path to your checkpoint. By default all  checkpoints are stored in checpoints folder:
```bash
python scripts/inference.py
```
For generation 10 motion at one time you can use bash script:
```
./generation.sh
```

### 5. Visualize them
Visualize generated or start motions using Viser. U can specify directory for generation via --motion_folder flag:
```bash
python scripts/viser_motion_play.py
```
