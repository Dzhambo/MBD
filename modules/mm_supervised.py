import torch


class MM_seq_encoder(torch.nn.Module):
    def __init__(self, seq_encoders):
        super().__init__()
        self.seq_encoders = torch.nn.ModuleDict(seq_encoders)

    def forward(self, x):
        res = []
        for source_name in self.seq_encoders:
            res.append(self.seq_encoders[source_name](x[source_name]))

        x = torch.cat(res, dim=-1)
        return x