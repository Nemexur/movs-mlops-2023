import torch


class Classification(torch.nn.Module):
    def __init__(self, in_features: int, num_classes: int) -> None:
        super().__init__()
        self._model = torch.nn.Sequential(
            torch.nn.Linear(in_features=in_features, out_features=100),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=100, out_features=num_classes),
        )
        self._loss = torch.nn.CrossEntropyLoss()

    def forward(self, inputs: dict[str, torch.Tensor]) -> None:
        logits = self._model(inputs["features"])
        output_dict = {"logits": logits, "probs": logits.softmax(dim=-1)}
        if (target := inputs.get("target")) is not None:
            output_dict["target"] = target
            output_dict["loss"] = self._loss(logits, target)
        return output_dict
