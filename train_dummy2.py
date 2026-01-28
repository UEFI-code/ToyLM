import EnDecoder
import LangModel
import torch
import torch.nn as nn
import dataset as dataset
from tqdm import tqdm
import gpu_chooser
import time

contextSize = 64
batchSize = 64
epoch = 1000
learning_rate, weight_decay = 1e-3, 0

trainingDevice = gpu_chooser.choose_gpu()
print(f'Training on device: {trainingDevice}')

datar = dataset.DataWarpper(contextSize, './demo_txt_dataset')

# Load Encoder-Decoder model
en_decoder = EnDecoder.EnDecoder(embeddingDim=4)
en_decoder.load_state_dict(torch.load('encoder_decoder.pth'))
en_decoder = en_decoder.to(trainingDevice)

langModel = LangModel.LangModel(max_seq_len=contextSize, embedding_dim=4, hidden_dim=64)
langModel = langModel.to(trainingDevice)

def infer_pipeline(source):
    source_encoded = en_decoder.pre_embedding(source)
    lang_latent = langModel(source_encoded)
    out = en_decoder.decoder(lang_latent)
    return out

def test(test_batch):
    assert test_batch.shape[0] == 1
    res = ''
    for _ in range(64):
        print(f'In: {test_batch}')
        out = infer_pipeline(test_batch)
        print(f'Out: {out}')
        byte = out.argmax(dim=-1).item()
        decoded_str = chr(byte)
        print(f'Decoded: {decoded_str}')
        res += decoded_str
        test_batch = torch.concat((test_batch[:, 1:], torch.tensor([[byte]], device=trainingDevice, dtype=torch.long)), dim=1)
    print(f'Final Result: {res}')

optim = torch.optim.SGD(langModel.parameters(), lr=learning_rate, weight_decay=weight_decay)
lossfunc = nn.CrossEntropyLoss()

source, target = datar.makeBatch(batchSize)
source = source.to(trainingDevice)
target = target.to(trainingDevice)

for n in tqdm(range(epoch)):
    optim.zero_grad()
    out = infer_pipeline(source)
    # calc loss
    loss = lossfunc(out, target)
    print(f'Epoch {n} Loss: {loss.item()}')
    loss.backward()
    optim.step()
    if n % 512 == 15:
        state_dict = langModel.state_dict()
        torch.save(state_dict, 'lang_model.pth')
        print('Model saved')

test(source[0:1])