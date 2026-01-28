import torch
import torch.nn as nn

class LangModel(nn.Module):
    def __init__(self, max_seq_len=64, embedding_dim=4, hidden_dim=64, depth=3):
        super(LangModel, self).__init__()
        self.mlp = nn.Sequential()
        self.mlp.append(nn.Linear(max_seq_len * embedding_dim, hidden_dim))
        self.mlp.append(nn.GELU())
        for _ in range(depth):
            self.mlp.append(nn.Linear(hidden_dim, hidden_dim))
            self.mlp.append(nn.GELU())
        self.mlp.append(nn.Linear(hidden_dim, embedding_dim))
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        return x

if __name__ == "__main__":
    x = torch.tensor([[[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]]])
    model = LangModel(max_seq_len=2, embedding_dim=4, hidden_dim=16)
    out = model.forward(x)
    print(out)