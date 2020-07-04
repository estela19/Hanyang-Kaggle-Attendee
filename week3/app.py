import torch


class SubmittedApp:
    def __init__(self):
        pass

    def run(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return input_tensor

    def metric(self, inferred_tensor: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        return torch.mean((inferred_tensor.argmax(dim=1) == ground_truth).float(), dim=-1)

