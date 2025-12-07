# >>>>>>>>
import torch
import numpy as np


def backtrack_reward(
    queries: list[torch.Tensor], responses: list[torch.Tensor], labels: list[torch.Tensor], backtrack_token_id: int
) -> torch.Tensor:
    rewards = []
    for query, response, label in zip(queries, responses, labels):
        verifier_reward, ce_reward, len_reward = 0.0, 0.0, 0.0

        src_num_bt_tk = (response == backtrack_token_id).sum()
        tar_num_bt_tk = (label == backtrack_token_id).sum()

        diff_bt_tk = tar_num_bt_tk - src_num_bt_tk

        if src_num_bt_tk == 0:
            verifier_reward -= 5.0  # Punishment for no backtracking

        verifier_reward += abs(diff_bt_tk) * 0.5

        diff_len = response.shape[0] - label.shape[0]
        len_reward += np.log(abs(diff_len) + 1) * 0.5

        reward = verifier_reward + ce_reward + len_reward
        rewards.append(reward)

    return rewards


REWARD_FUNCTIONS = {
    "backtrack": backtrack_reward,
}


def load_custom_reward(target: str) -> callable:
    if target in REWARD_FUNCTIONS:
        return REWARD_FUNCTIONS[target]
    else:
        raise ValueError(f"Custom reward function {target} not found.")


# <<<<<<<<
