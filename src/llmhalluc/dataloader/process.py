from datasets import Dataset, DatasetDict, load_dataset
from typing import Callable, Any
from pathlib import Path
from functools import partial

from transformers import AutoTokenizer


from llmhalluc.data.backtrack import backtrack_processor, BACKTRACK_TOKEN


def process_dataset(
    dataset: Dataset | DatasetDict,
    processor: Callable,
    dataset_name: str,
    data_dir: str = "./data",
    split: str | list[str] = "train",
    repeat: int = 1,
    dataset_kwargs: dict[str, Any] = {"batched": False, "num_proc": 12},
):
    if isinstance(split, list):
        for split in split:
            process_dataset(dataset, processor, dataset_name, data_dir, split, repeat, dataset_kwargs)
    else:
        if isinstance(dataset, DatasetDict):
            dataset = dataset[split]
        dataset = dataset.repeat(repeat)
        column_names = dataset.column_names
        processed_dataset = dataset.map(processor, remove_columns=column_names, **dataset_kwargs)
        dataset_name = f"{dataset_name}_repeat_{repeat}"
        save_path = Path(data_dir) / dataset_name / f"{split}.json"
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True, exist_ok=True)
        processed_dataset.to_json(str(save_path), orient="records")


split, repeat = "validation", 1
dataset = load_dataset("json", data_files=f"./data/squad_v2/{split}.json")["train"]
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507", cache_dir="./.cache", trust_remote_code=True)
tokenizer.add_tokens([BACKTRACK_TOKEN], special_tokens=True)
vocab = set(tokenizer.get_vocab().values())
special_tokens = set(tokenizer.all_special_ids)
no_spc_vocab = list(vocab - special_tokens)
processor = partial(backtrack_processor, tokenizer=tokenizer, no_spc_vocab=no_spc_vocab, split=split)

process_dataset(dataset, processor, dataset_name="squad_v2_backtrack", data_dir="./data", split=split, repeat=repeat)
