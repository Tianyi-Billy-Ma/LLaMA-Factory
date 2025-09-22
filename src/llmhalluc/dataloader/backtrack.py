from typing import Any

import numpy as np
from transformers import PreTrainedTokenizer

BACKTRACK_TOKEN = "<|BACKTRACK|>"


def backtrack_processor(
    example: dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    max_tokens: int = 10,
    key_mapping: dict[str, str] = {"prompt": "prompt", "query": "input", "response": "answer"},
    no_spc_vocab: list[int] = None,
    split: str = "train",
) -> dict[str, Any]:
    prompt = example[key_mapping["prompt"]]
    query = example[key_mapping["query"]]
    response = example[key_mapping["response"]]
    response_token_ids = tokenizer.encode(response)

    backtrack_id = tokenizer.encode(BACKTRACK_TOKEN)[0]

    if not no_spc_vocab:
        vocab = set(tokenizer.get_vocab().values())
        special_tokens = set(tokenizer.all_special_ids)
        no_spc_vocab = list(vocab - special_tokens)

    random_int = np.random.randint(0, max_tokens) if split == "train" else 0
    random_split = np.random.randint(0, len(response_token_ids))
    # If no wrong tokens, return original example
    if random_int == 0:
        return {
            "prompt": prompt,
            "query": query,
            "response": response,
            "backtrack_content": "",
        }
    np.random.shuffle(no_spc_vocab)
    backtrack_token_ids = response_token_ids[:random_split] + no_spc_vocab[:random_int]
    curr_response_token_ids = [backtrack_id] * random_int + response_token_ids[random_split:]

    # Count how many tokens the backtrack_content contains when tokenized
    backtrack_content = tokenizer.decode(backtrack_token_ids)
    modified_response = tokenizer.decode(curr_response_token_ids)

    return {
        "prompt": prompt,
        "query": query,
        "response": modified_response,
        "backtrack_content": backtrack_content,
    }
