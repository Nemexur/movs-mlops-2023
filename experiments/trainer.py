from typing import Any, Callable

from accelerate import Accelerator
from ignite.engine import Engine, Events, State
import torch
from torch.utils.data import DataLoader


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        accelerator: Accelerator,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.engines = {"train": Engine(self._train_step), "eval": Engine(self._eval_step)}
        self._accelerator = accelerator
        self._add_events()
        for key, e in self.engines.items():
            e.state.name = key
            e.state.epoch_iteration = 0
            e.state_dict_user_keys.append("name")
            e.state_dict_user_keys.append("epoch_iteration")

    def add_event(
        self, engine: str, event_name: Any, handler: Callable, *args: Any, **kwargs: Any
    ) -> None:
        self.engines[engine].add_event_handler(event_name, handler, *args, **kwargs)

    def run(
        self, loaders: dict[str, DataLoader], max_iters: dict[str, int], epochs: int | None = None
    ) -> State:
        self._loaders = loaders
        self._max_iters = max_iters
        self.engines["train"].run(
            self._loaders["train"],
            epoch_length=self._max_iters.get("train"),
            max_epochs=epochs,
        )
        return self.engines["eval"].state if "eval" in loaders else self.engines["train"].state

    def _train_step(
        self, engine: Engine, batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        self.model.train()
        with self._accelerator.accumulate(self.model):
            state = engine.state
            output = state.output = self.model(batch)
            if "loss" not in output:
                return output
            self._accelerator.backward(output["loss"])
            self.optimizer.step()
            self.optimizer.zero_grad()
            state.metrics["_loss"] += output["loss"].detach()
            return output

    def _eval_step(self, engine: Engine, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self.model.eval()
        with torch.no_grad():
            state = engine.state
            output = state.output = self.model(batch)
            if "loss" in output:
                state.metrics["_loss"] += output["loss"].detach()
            return output

    def _add_events(self) -> None:
        self.add_event("train", Events.EPOCH_STARTED | Events.COMPLETED, self._run_eval)
        events = (
            (Events.EPOCH_STARTED, self._reset_epoch),
            (Events.ITERATION_COMPLETED, self._update_iteration),
            (Events.ITERATION_COMPLETED, self._update_loss),
        )
        for e in self.engines:
            for args in events:
                self.add_event(e, *args)

    def _run_eval(self) -> None:
        eval_loader = self._loaders.get("eval")
        if eval_loader is None:
            return
        self.engines["eval"].run(eval_loader, epoch_length=self._max_iters.get("eval"))

    def _reset_epoch(self, engine: Engine) -> None:
        state = engine.state
        state.metrics["_loss"] = torch.tensor(0.0, device=self._accelerator.device)
        state.epoch_iteration = 0

    def _update_iteration(self, engine: Engine) -> None:
        engine.state.epoch_iteration += 1

    def _update_loss(self, engine: Engine) -> None:
        state = engine.state
        state.metrics["loss"] = state.metrics["_loss"] / state.epoch_iteration
