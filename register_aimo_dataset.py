"""
Register AI-MO/NuminaMath-CoT as the dataset name "aimo" for rllm.

This script is intended to be called once before launching the training job.
"""

from datasets import load_dataset
from rllm.data.dataset import DatasetRegistry


def main() -> None:
    split = "train"
    ds = load_dataset("AI-MO/NuminaMath-CoT", split=split)

    def to_row(ex):
        return {
            "question": ex.get("question") or ex.get("problem") or ex.get("input") or ex.get("prompt"),
            "ground_truth": ex.get("ground_truth") or ex.get("answer") or ex.get("output") or ex.get("final_answer"),
            "data_source": "aimo",
        }

    ds = ds.map(to_row)
    DatasetRegistry.register_dataset("aimo", ds, split)
    print("registered aimo", split, len(ds))


if __name__ == "__main__":
    main()
