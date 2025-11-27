from datasets import load_dataset
from rllm.data.dataset import DatasetRegistry


def to_row(ex):
    return {
        "question": ex.get("problem") or ex.get("question") or ex.get("input"),
        "ground_truth": ex.get("solution") or ex.get("answer") or ex.get("output"),
        "data_source": "math500",
    }


def register(split: str, hf_split: str = "test"):
    ds = load_dataset("HuggingFaceH4/MATH-500", split=hf_split)
    ds = ds.map(to_row)
    DatasetRegistry.register_dataset("math500", ds, split)
    print(f"registered math500 split={split} (from hf_split={hf_split}) n={len(ds)}")


if __name__ == "__main__":
    # Register under both 'train' (for convenience) and 'test' aliases.
    register("train", hf_split="test")
    register("test", hf_split="test")
