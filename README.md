# Gen–Ver DAgger (vLLM Full-FT)

Minimal instructions for bringing up the generator–verifier DAgger pipeline.

## Setup
```bash
conda create -y -n rllm python=3.10
conda activate rllm
python -m pip install --upgrade pip
pip install -r requirements.txt
```

- `requirements.txt` reuses `rllm/verl/requirements.txt`, installs `rllm` in editable mode, and pins
  `vllm==0.11.0` (matching the recorded rllm env in `MADAgger/wandb/run-20251116_100911-qp3ggvk9/files/requirements.txt`).
- Install optional extras as needed, e.g. `pip install -r rllm/docs/requirements.txt`.

## Usage
Roll alternating generator/verifier rounds:
```bash
python MADAgger/gen_ver_dagger_fullft_vllm.py iterate \
  --dataset_name deepscaler_math --split train --rounds 5 --batch_tasks 256 \
  --teacher_base Qwen/Qwen3-4B --gen_base Qwen/Qwen3-0.6B --ver_base Qwen/Qwen3-0.6B \
  --out_dir runs/genver_fullft --project_name genver_fullft --experiment_name exp \
  --parallel 64 --tp_s 1 --tp_t 1 --gpu_mem_util 0.9 --train_cuda "0,1"
```

Collect SFT rows without training:
```bash
python MADAgger/gen_ver_dagger_fullft_vllm.py collect \
  --dataset_name deepscaler_math --split train --num_tasks 512 \
  --teacher_base Qwen/Qwen3-4B --gen_base Qwen/Qwen3-0.6B --ver_base Qwen/Qwen3-0.6B \
  --save_prefix datasets/collect --parallel 64
```

## Notes
- Keep teacher/gen/verifier vLLM engines on disjoint GPUs; overlap only if you accept slower runs.
- Use `--train_inline` to run VERL SFT in-process; the default launches a `torchrun` subprocess.
- Generated SFT parquet files accumulate under `--out_dir` and are deduplicated automatically.
