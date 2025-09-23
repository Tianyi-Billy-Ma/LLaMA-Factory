from datasets import Dataset, DatasetDict
from typing import Callable, Any
from pathlib import Path


def process_dataset(
    dataset: Dataset | DatasetDict,
    processor: Callable,
    dataset_name: str,
    data_dir: str = "./data",
    split: str | list[str] = "train",
    repeat: int = 1,
    dataset_kwargs: dict[str, Any] = {"batched": False, "num_proc": 12},
    **kwargs,
):
    dataset_kwargs.update(kwargs)
    if isinstance(split, list):
        for split in split:
            process_dataset(dataset, processor, dataset_name, data_dir, split, repeat, dataset_kwargs)
    else:
        if isinstance(dataset, DatasetDict):
            dataset = dataset[split]
        dataset = dataset.repeat(repeat)
        column_names = dataset.column_names
        processed_dataset = dataset.map(processor, remove_columns=column_names, **dataset_kwargs)
        save_path = Path(data_dir) / dataset_name / f"{split}.json"
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True, exist_ok=True)
        processed_dataset.to_json(str(save_path), orient="records")
