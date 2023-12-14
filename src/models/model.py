import torch


class Classification(torch.nn.Module):
    def __init__(self, in_features: int, num_classes: int, hidden_dim: int = 100) -> None:
        super().__init__()
        self._model = torch.nn.Sequential(
            torch.nn.Linear(in_features=in_features, out_features=hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=hidden_dim, out_features=num_classes),
        )
        self._loss = torch.nn.CrossEntropyLoss()

    def forward(self, inputs: dict[str, torch.Tensor]) -> None:
        logits = self._model(inputs["features"])
        output_dict = {"logits": logits, "probs": logits.softmax(dim=-1)}
        if (target := inputs.get("target")) is not None:
            output_dict["target"] = target
            output_dict["loss"] = self._loss(logits, target)
        return output_dict
