from setuptools import setup

setup(
    name="madagger",
    version="0.1.0",
    description="Gen-Ver DAgger pipeline with vLLM full fine-tuning",
    py_modules=[
        "checkpoint_utils",
        "data_utils",
        "fullft_training",
        "gen_ver_dagger_fullft_vllm",
        "genver_teacher",
        "genver_workflow",
        "gpu_utils",
        "vllm_engine",
    ],
    install_requires=[],
    python_requires=">=3.10",
)
