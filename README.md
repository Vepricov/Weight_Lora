# ``Wandb`` project: 
https://wandb.ai/shkodnik-mipt/SBER_LORA?nw=nwusershkodnik

# Math Tutor project
This project aims to create an AI tutor, that can help student to solve math problems using [socratic method](https://en.wikipedia.org/wiki/Socratic_method).
High-level requirements for this tutor are: helps to solve by asking questions that stimulate reflection; doesn't leak answer or solve problem instead of student; its questions should guide the student toward discovering the solution independently; breaks down complex problems into smaller, manageable steps; capable of detecting valid solution steps and correcting mistakes.
Currently tutoring skills are learned from semi-synthetic dataset [MathDial](https://github.com/eth-nlped/mathdial), translated to russian language by GPT4-o.

## Requirements 
    python 3.11.7

## Dependancies installation
Plese follow these steps (required!):

0. If you don't have python of specific version:
    - Install conda
    - Run:
      ```bash
      conda create --name python3.11.7 python=3.11.7
      conda activate python3.11.7
      ```

1. Create virtual env
```bash
python3.11 -m venv venv
source venv/bin/activate
```

2. Update pip to latest
```bash
pip install pip==24.1.1
```
3. Install Unsloth package

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

4. Install rest of packages from requirements.txt
```bash
pip install -r requirements.txt
```

## Usage
### Train Scripts
Up to this point, experiments were based on Meta-Llama-3.1-8B from unsloth library (see https://huggingface.co/unsloth/Meta-Llama-3.1-8B)

1. SFT
QLORA finetune training:

```bash
# CUDA_VISIBLE_DEVICES -- to select a specific GPU on the server 
# P.S. on MIPT server number 2 is A100
CUDA_VISIBLE_DEVICES=2 python pipelines/unsloth_qlora_finetune.py
```
