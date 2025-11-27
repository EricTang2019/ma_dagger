# Gen–Ver DAgger (vLLM Full-FT)

Minimal instructions for bringing up the generator–verifier DAgger pipeline.

## Setup
```bash
conda create -y -n madagger python=3.10
conda activate madagger
python -m pip install --upgrade pip
pip install -r requirements.txt

# Install the upstream rLLM dependency (clone inside this project; install manually to keep Torch/vLLM pinned).
git clone --recurse-submodules https://github.com/rllm-org/rllm.git deps/rllm
pushd deps/rllm
pip install -e .
# Install the Verl python package from the same repo (this command must be run inside deps/rllm).
pip install --no-deps -e verl
popd

# Finally install this repo
pip install -e .
```
- `requirements.txt` lists only the pip-installable Python packages (rLLM 0.2.0, VERL extras, Torch 2.8, vLLM 0.11, etc.).
  CUDA toolkits, `_libgcc_mutex`, and other Conda/NVIDIA components still need to come from the conda env you create
  in the first two commands.
- `flash-attn>=2.7.4` is included for Verl; if pip hits a build error, rerun `pip install "flash-attn>=2.7.4" --no-build-isolation`.
- Install optional extras as needed, e.g. `pip install -r rllm/docs/requirements.txt`.
- Make sure the dataset you reference (e.g. `gsm8k`, `deepscaler_math`) has been registered with
  `DatasetRegistry` via rLLM’s data-prep scripts.

## Usage
Roll alternating generator/verifier rounds:
```bash
python gen_ver_dagger_fullft_vllm.py iterate \
  --dataset_name deepscaler_math --split train --rounds 5 --batch_tasks 256 \
  --teacher_base Qwen/Qwen3-4B --gen_base Qwen/Qwen3-0.6B --ver_base Qwen/Qwen3-0.6B \
  --out_dir runs/genver_fullft --project_name genver_fullft --experiment_name exp \
  --parallel 64 --tp_s 1 --tp_t 1 --gpu_mem_util 0.9 --train_cuda "0,1"
```

Collect SFT rows without training:
```bash
python gen_ver_dagger_fullft_vllm.py collect \
  --dataset_name deepscaler_math --split train --num_tasks 512 \
  --teacher_base Qwen/Qwen3-4B --gen_base Qwen/Qwen3-0.6B --ver_base Qwen/Qwen3-0.6B \
  --save_prefix datasets/collect --parallel 64
```

## Notes
- Keep teacher/gen/verifier vLLM engines on disjoint GPUs; overlap only if you accept slower runs.
- Use `--train_inline` to run VERL SFT in-process; the default launches a `torchrun` subprocess.
- Generated SFT parquet files accumulate under `--out_dir` and are deduplicated automatically.





export AZURE_OPENAI_AD_TOKEN="$(az account get-access-token --resource api://trapi --query accessToken -o tsv)" \
&& python launch_azure_ma_dagger.py



