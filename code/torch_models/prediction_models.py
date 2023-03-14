from typing import Optional
import torch


class PredictionModelInterface:
    def __init__(self): pass

    def predict(self, inp: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class LSTM(PredictionModelInterface):
    def __init__(self, **lstm_args):
        super().__init__()

        self.model = torch.nn.LSTM(**lstm_args)

    def forward(self,
                inp: torch.Tensor,
                h0: Optional[torch.Tensor] = None,
                c0: Optional[torch.Tensor] = None
                ) -> tuple:
        # inp.shape = (seq_len, inp_dim)
        # h0.shape = (num_layers, hid_dim)
        # co.shape = (num_layers, hid_dim)
        if h0 is None:
            hid_size = (
                (self.model.bidirectional + 1) * self.model.num_layers,
                self.model.hidden_size if self.model.proj_size == 0 \
                else self.model.proj_size
            )
            h0 = torch.zeros(hid_size).to(inp.device)

        if c0 is None:
            cell_size = (
                (self.model.bidirectional + 1) * self.model.num_layers,
                self.model.hidden_size
            )
            c0 = torch.zeros(cell_size).to(inp.device)

        return self.model(inp, (h0, c0))

    def predict(self, inp: torch.Tensor) -> torch.Tensor:
        # inp.shape = (seq_len, inp_dim)
        # out.shape = (seq_len, proj_size or hid_dim)
        self.model.eval()

        with torch.no_grad():
            out, (hn, cn) = self.forward(inp)

        return out
