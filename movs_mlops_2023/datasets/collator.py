from typing import Any
from collections import defaultdict

import torch
from torch.nn.utils.rnn import pad_sequence


class Default:
    def __init__(self, pad: list[str] | None = None, padding_value: float = 0) -> None:
        self._pad = set(pad or [])
        self._padding_value = padding_value

    def __call__(self, instances: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch = {
            key: pad_sequence(
                [torch.as_tensor(t) for t in tensor],
                batch_first=True,
                padding_value=self._padding_value,
            )
            if key in self._pad
            else torch.tensor(tensor)
            for key, tensor in self._make_batch(instances).items()
        }
        for key in self._pad:
            batch[f"{key}_mask"] = batch[key].ne(self._padding_value).float()
        return batch

    @staticmethod
    def _make_batch(instances: list[dict[str, Any]]) -> dict[str, list[Any]]:
        tensor_dict = defaultdict(list)
        for instance in instances:
            for field, tensor in instance.items():
                tensor_dict[field].append(tensor)
        return tensor_dict
