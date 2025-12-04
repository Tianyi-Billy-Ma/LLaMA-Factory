# >>>>>>>>
import importlib
import importlib.util
import inspect
import os
import uuid
from types import ModuleType
from typing import Any, Callable, Iterable, Sequence

import torch

CallableRewards = Callable[[Sequence[torch.Tensor], Sequence[torch.Tensor]], torch.Tensor]


def _import_module(module_path: str) -> ModuleType:
    if os.path.isfile(module_path):
        module_name = f"llamafactory_custom_reward_{uuid.uuid4().hex}"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from path: {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[arg-type]
        return module

    return importlib.import_module(module_path)


def _resolve_target(target: str) -> tuple[ModuleType, str]:
    module_path, sep, attr_name = target.partition(":")
    if not sep or not attr_name:
        raise ValueError(
            "Custom reward path must be in the form 'path.to.module:callable' or '/path/file.py:callable'."
        )
    module = _import_module(module_path)
    if not hasattr(module, attr_name):
        raise AttributeError(f"Attribute '{attr_name}' not found in module '{module_path}'.")
    return module, attr_name


def _as_tensor(output: Any, expected: int) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        tensor = output.detach().cpu().to(torch.float32).view(-1)
    elif isinstance(output, Iterable):
        values: list[torch.Tensor] = []
        for item in output:
            if isinstance(item, torch.Tensor):
                values.append(item.detach().cpu().to(torch.float32).view(-1)[0])
            else:
                values.append(torch.tensor(item, dtype=torch.float32))
        tensor = torch.stack(values).view(-1)
    else:
        raise TypeError("Custom reward callable must return a Tensor or iterable of scalars.")

    if tensor.numel() != expected:
        raise ValueError(f"Custom reward must return {expected} values, got {tensor.numel()}.")

    return tensor


def load_custom_reward(target: str) -> CallableRewards:
    module, attr_name = _resolve_target(target)
    reward_obj = getattr(module, attr_name)
    if inspect.isclass(reward_obj):
        reward_obj = reward_obj()

    if not callable(reward_obj):
        raise TypeError(f"Object '{attr_name}' in '{module.__name__}' is not callable.")

    def reward_fn(queries: Sequence[torch.Tensor], responses: Sequence[torch.Tensor]) -> torch.Tensor:
        rewards = reward_obj(queries, responses)
        return _as_tensor(rewards, len(queries))

    return reward_fn
# <<<<<<<<

