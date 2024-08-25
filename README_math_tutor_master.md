# Math Tutor project
This project aims to create an AI tutor, that can help student to solve math problems using [socratic method](https://en.wikipedia.org/wiki/Socratic_method).
High-level requirements for this tutor are: helps to solve by asking questions that stimulate reflection; doesn't leak answer or solve problem instead of student; its questions should guide the student toward discovering the solution independently; breaks down complex problems into smaller, manageable steps; capable of detecting valid solution steps and correcting mistakes.
Currently tutoring skills are learned from semi-synthetic dataset [MathDial](https://github.com/eth-nlped/mathdial), translated to russian language by GPT4-o.

### Requirements
    python 3.11.7

### Dependancies installation
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
3. Install packages for Unsloth separately

```bash
pip install --upgrade --force-reinstall --no-cache-dir torch==2.2.0 triton==2.2.0 \
  --index-url https://download.pytorch.org/whl/cu121
pip install wheel==0.43.0 packaging==24.1
pip install flash-attn==2.5.9.post1 --no-build-isolation
pip install "unsloth[cu121-ampere-torch220] @ git+https://github.com/unslothai/unsloth.git"
```

4. Install rest of packages from requirements.txt
```bash
pip install -r requirements.txt
```

### Usage
#### Preprocess Scripts


#### Train/Inference Scripts
Up to this point, experiments were based on either llama3-8b-instruct and llama3.1-8b-instruct

To implement training pipeline, use these scripts in following order (check out non-default arguments in all scripts):
1. SFT
LORA (Full precision) finetune training:
```bash
CUDA_VISIBLE_DEVICES=1 python pipeline/unsloth_lora_finetune.py
```
QLORA finetune training:

```bash
CUDA_VISIBLE_DEVICES=1 python pipeline/unsloth_qlora_finetune.py
```

2. DPO:
```bash
CUDA_VISIBLE_DEVICES=1 python pipeline/unsloth_dpo.py
```
3. Inference and saving model outputs to file:
```bash
CUDA_VISIBLE_DEVICES=1 python pipeline/unsloth_inference.py --model_name unsloth/llama-3-8b-Instruct-bnb-4bit --adapter_path /media/ssd-5t/eshevtsova/sft/llama3-math-dial/checkpoint-8500
```
3. Evaluation
A. Simularity-based:
pass model(s) output file(s); will output metrics comparison table and save it so csv:
```bash
CUDA_VISIBLE_DEVICES=1 python evaluate.py data/test_dpo_llama3.jsonl data/test_dpo_llama3_ft.jsonl data/test_dpo_llama3_dpo.jsonl --batch_size 1024
```
B. CAR@k
```bash
CUDA_VISIBLE_DEVICES=1 python pipeline/evaluate_dialog.py
```