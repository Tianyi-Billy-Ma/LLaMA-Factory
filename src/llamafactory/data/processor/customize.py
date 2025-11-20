# >>>>>>>>
from dataclasses import dataclass
from typing import Any, Optional

from .processor_utils import DatasetProcessor


DEFAULT_REPLACE_COLUMNS = ["_prompt", "_response", "_system"]


def _replace_text_examples(
    examples: dict[str, list[Any]], replace_text: Optional[dict[str, str]]
) -> dict[str, list[Any]]:
    if replace_text is None:
        return examples

    # Apply all replacements in a single pass per column
    for target_column in DEFAULT_REPLACE_COLUMNS:
        if target_column not in examples:
            continue

        examples[target_column] = [
            _apply_replacements(example, replace_text) if isinstance(example, str) else example
            for example in examples[target_column]
        ]

    return examples


def _apply_replacements(text: str, replace_text: dict[str, str]) -> str:
    """Apply all replacements to a single text string."""
    result = text
    for source_text, target_text in replace_text.items():
        result = result.replace(source_text, target_text)
    return result


@dataclass
class CustomizeDatasetProcessor(DatasetProcessor):
    base_class: type[DatasetProcessor]

    def __post_init__(self):
        self.base_processor = self.base_class(
            template=self.template,
            tokenizer=self.tokenizer,
            processor=self.processor,
            data_args=self.data_args,
        )

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        examples = _replace_text_examples(examples, self.data_args.replace_text)
        return self.base_processor.preprocess_dataset(examples)

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        self.base_processor.print_data_example(example)


# <<<<<<<<
