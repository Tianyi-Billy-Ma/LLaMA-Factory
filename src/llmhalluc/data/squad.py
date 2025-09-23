"""Squad dataset converter."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset

from .base import DatasetConverter
from ..prompts.QAPrompt import QA_INSTRUCTION


@dataclass
class SquadDatasetConverter(DatasetConverter):
    """Converter for Squad v2 dataset.

    Converts Squad v2 examples to the standard format with prompt, query, and response.

    Args:
        cache_dir: Directory for caching downloaded datasets.
        data_dir: Directory for saving processed datasets.
        redownload: Whether to force redownload the dataset.
    """

    redownload: bool = False

    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        """Convert a Squad v2 example to standard format.

        Args:
            example: Squad v2 example with 'context', 'question', and 'answers' keys.

        Returns:
            Converted example with 'prompt', 'query', and 'response' keys.
        """
        return {
            "prompt": QA_INSTRUCTION,
            "query": f"Context: {example['context']}\nQuestion: {example['question']}",
            "response": (example["answers"]["text"][0] if len(example["answers"]["text"]) > 0 else "IDK"),
        }

    def load_and_process_dataset(self, splits: str | list[str] = ["train", "validation"]) -> None:
        """Load and process Squad v2 dataset.

        Args:
            splits: Dataset splits to process.
        """
        if isinstance(splits, str):
            splits = [splits]

        for split in splits:
            # Load dataset
            data = load_dataset(
                "rajpurkar/squad_v2",
                cache_dir=self.cache_dir,
                split=split,
                download_mode="force_redownload" if self.redownload else "reuse_dataset_if_exists",
            )

            # Process dataset
            dataset = data.map(self, batched=False, remove_columns=data.column_names)

            # Save processed dataset
            save_path = Path(self.data_dir) / "squad_v2" / f"{split}.json"
            if not save_path.parent.exists():
                save_path.parent.mkdir(parents=True, exist_ok=True)

            dataset.to_json(str(save_path), orient="records")
