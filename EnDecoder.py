import torch
import torch.nn as nn

class EnDecoder(nn.Module):
    def __init__(self, embeddingDim = 4):
        super().__init__()
        self.pre_embedding = nn.Embedding(256, embeddingDim)
        self.decoder = nn.Linear(embeddingDim, 256)

    def forward(self, x):
        x = self.pre_embedding(x)
        x = self.decoder(x)
        return x

if __name__ == "__main__":
    x = torch.tensor([[i for i in range(256)]])
    model = EnDecoder()
    print(model.forward(x))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    for _ in range(9999):
        out = model.forward(x)
        # calc cross-entropy loss between out and x
        # before calc cross-entropy, we need merge the tensor to (batch_size * seq_len, 256)
        loss = loss_fn(out.view(-1, 256), x.view(-1))
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # show output with argmax
    print(out.argmax(dim=-1))
    torch.save(model.state_dict(), 'encoder_decoder.pth')