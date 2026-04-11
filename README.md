# Text-to-Motion project

This is project for generating reference motions on G1 Unitree humanoid robot using text prompts

## Installation & Setup

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

## Launch Training
```bash
python scripts/train.py
```

## Generate Reference
```bash
python scripts/inference.py
```

## Visualize them
```bash
python scripts/viser_motion_play.py
```