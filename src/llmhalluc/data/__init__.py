"""Dataset converters for llmhalluc."""

from .base import DatasetConverter
from .squad import SquadDatasetConverter
from .backtrack import BacktrackDatasetConverter, BACKTRACK_TOKEN

# Registry of available converters
DATASET_CONVERTERS = {
    "squad": SquadDatasetConverter,
    "backtrack": BacktrackDatasetConverter,
}


def register_dataset_converter(name: str, converter_class: type[DatasetConverter]) -> None:
    """Register a new dataset converter.

    Args:
        name: Name of the converter.
        converter_class: Converter class to register.

    Raises:
        ValueError: If converter name already exists.
    """
    if name in DATASET_CONVERTERS:
        raise ValueError(f"Dataset converter {name} already exists.")

    DATASET_CONVERTERS[name] = converter_class


def get_dataset_converter(name: str, **kwargs) -> DatasetConverter:
    """Get a dataset converter instance.

    Args:
        name: Name of the converter.
        **kwargs: Arguments to pass to the converter constructor.

    Returns:
        Converter instance.

    Raises:
        ValueError: If converter name not found.
    """
    if name not in DATASET_CONVERTERS:
        raise ValueError(f"Dataset converter {name} not found.")

    return DATASET_CONVERTERS[name](**kwargs)


__all__ = [
    "DatasetConverter",
    "SquadDatasetConverter",
    "BacktrackDatasetConverter",
    "BACKTRACK_TOKEN",
    "DATASET_CONVERTERS",
    "register_dataset_converter",
    "get_dataset_converter",
]
