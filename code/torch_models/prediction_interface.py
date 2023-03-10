import torch


class PredictionModelInterface:
    def __init__(self): pass

    def predict(self, inp: torch.Tensor, target: torch.Tensor):
        raise NotImplementedError()