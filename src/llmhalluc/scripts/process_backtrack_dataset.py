import argparse
from typing import Callable
from pathlib import Path
from functools import partial

from transformers import AutoTokenizer, PreTrainedTokenizer
from datasets import load_dataset
from llmhalluc.dataloader import process_dataset, PROCESSOR_FN_MAP, BACKTRACK_TOKEN


def process_backtrack_dataset(
    dataset_name: str,
    fn: Callable,
    tokenizer: PreTrainedTokenizer,
    no_spc_vocab: list[int],
    data_dir: str = "./data",
    split: str = "train",
    repeat: int = 1,
    num_proc: int = 12,
):
    repeat = repeat if split == "train" else 1
    input_path = Path(data_dir) / dataset_name / f"{split}.json"
    dataset = load_dataset("json", data_files=str(input_path))["train"]
    processor = partial(fn, tokenizer=tokenizer, no_spc_vocab=no_spc_vocab, split=split)
    process_dataset(
        dataset,
        processor,
        dataset_name=f"{dataset_name}_backtrack",
        data_dir=data_dir,
        split=split,
        repeat=repeat,
        num_proc=num_proc,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="./.cache")
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--dataset_name", default="squad_v2")
    parser.add_argument("--processor", default="backtrack")
    parser.add_argument("--split", default=["train", "validation"])
    parser.add_argument("--repeat", default=5)
    parser.add_argument("--num_proc", default=12)
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-4B-Instruct-2507")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, cache_dir=args.cache_dir, trust_remote_code=True)
    tokenizer.add_tokens([BACKTRACK_TOKEN], special_tokens=True)
    vocab = set(tokenizer.get_vocab().values())
    special_tokens = set(tokenizer.all_special_ids)
    no_spc_vocab = list(vocab - special_tokens)
    processer_fn = PROCESSOR_FN_MAP[args.processor]

    if isinstance(args.split, list):
        for split in args.split:
            process_backtrack_dataset(
                dataset_name=args.dataset_name,
                fn=processer_fn,
                tokenizer=tokenizer,
                no_spc_vocab=no_spc_vocab,
                data_dir=args.data_dir,
                split=split,
                repeat=args.repeat,
                num_proc=args.num_proc,
            )
    elif isinstance(args.split, str):
        process_backtrack_dataset(
            dataset_name=args.dataset_name,
            fn=processer_fn,
            tokenizer=tokenizer,
            no_spc_vocab=no_spc_vocab,
            data_dir=args.data_dir,
            split=args.split,
            repeat=args.repeat,
            num_proc=args.num_proc,
        )
    else:
        raise ValueError(f"Invalid split: {args.split}")


if __name__ == "__main__":
    main()
