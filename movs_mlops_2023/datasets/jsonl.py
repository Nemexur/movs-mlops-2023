from typing import Any, Iterator
from itertools import islice
import json
from pathlib import Path

from torch.utils.data import Dataset, IterableDataset, get_worker_info


class InMemory(Dataset):
    def __init__(self, path: Path | str) -> None:
        with Path(path).open("r", encoding="utf-8") as file:
            self._samples = [json.loads(line) for line in file]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._samples[idx]


class Iter(IterableDataset):
    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker_info = get_worker_info()
        with self._path.open("r", encoding="utf-8") as file:
            start, step = 0, 1
            if worker_info is not None and worker_info.num_workers > 0:
                start, step = worker_info.id, worker_info.num_workers
            lines = islice(file, start, None, step)
            yield from map(json.loads, lines)
