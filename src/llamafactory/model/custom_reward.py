# >>>>>>>>
import torch


def backtrack_reward(
    queries: list[torch.Tensor], responses: list[torch.Tensor], backtrack_token_id: int
) -> torch.Tensor:
    backtrack_token_id = torch.tensor(backtrack_token_id, dtype=torch.long)
    return torch.tensor([0.0] * len(queries))


REWARD_FUNCTIONS = {
    "backtrack": backtrack_reward,
}


def load_custom_reward(target: str) -> callable:
    if target in REWARD_FUNCTIONS:
        return REWARD_FUNCTIONS[target]
    else:
        raise ValueError(f"Custom reward function {target} not found.")


# <<<<<<<<
