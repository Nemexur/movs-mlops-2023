from typing import Any, Iterator
from itertools import islice
import json
from pathlib import Path

import dvc.api
from torch.utils.data import IterableDataset, get_worker_info


class Iter(IterableDataset):
    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker_info = get_worker_info()
        with dvc.api.open(str(self._path), mode="r", encoding="utf-8") as file:
            start, step = 0, 1
            if worker_info is not None and worker_info.num_workers > 0:
                start, step = worker_info.id, worker_info.num_workers
            lines = islice(file, start, None, step)
            yield from map(json.loads, lines)
