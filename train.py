import EnDecoder
import LangModel
import torch
import torch.nn as nn
import dataset as dataset
from tqdm import tqdm
import gpu_chooser
import conf
# import random

trainingDevice = gpu_chooser.choose_gpu()
print(f'Training on device: {trainingDevice}')

datar = dataset.DataWarpper(conf.contextSize+1, './demo_txt_dataset')

# Load Encoder-Decoder model
en_decoder = EnDecoder.EnDecoder(embeddingDim=conf.embedding_dim)
en_decoder.load_state_dict(torch.load('encoder_decoder.pth'))
en_decoder = en_decoder.to(trainingDevice)

langModel = LangModel.LangModel(max_seq_len=conf.contextSize, embedding_dim=conf.embedding_dim, hidden_dim=conf.hidden_dim, depth=conf.depth)
try:
    langModel.load_state_dict(torch.load('lang_model.pth'))
    print('Lang model loaded')
except:
    print('Lang model not found, initializing from scratch')
langModel = langModel.to(trainingDevice)

def infer_pipeline(source):
    source_encoded = en_decoder.pre_embedding(source)
    return langModel(source_encoded)

def test(test_batch):
    assert test_batch.shape[0] == 1
    res = ''
    for _ in range(conf.contextSize):
        print(f'In: {test_batch}')
        out = infer_pipeline(test_batch)
        print(f'Out: {out}')
        byte = out.argmax(dim=-1).item()
        decoded_str = chr(byte)
        print(f'Decoded: {decoded_str}')
        res += decoded_str
        test_batch = torch.concat((test_batch[:, 1:], torch.tensor([[byte]], device=trainingDevice, dtype=torch.long)), dim=1)
    print(f'Final Result: {res}')

optim = torch.optim.SGD(langModel.parameters(), lr=conf.learning_rate, weight_decay=conf.weight_decay)
lossfunc = nn.CrossEntropyLoss()

# def glicher(x):
#     for _ in range(x.size(0) // 2):
#         # trigger mutation
#         batch_id_a = random.randint(0, x.size(0) - 1)
#         pos_s = random.randint(0, x.size(1))
#         pos_e = random.randint(0, x.size(1))
#         batch_id_b = random.randint(0, x.size(0) - 1)
#         # swap
#         backup = x[batch_id_a, pos_s:pos_e].clone()
#         x[batch_id_a, pos_s:pos_e] = x[batch_id_b, pos_s:pos_e]
#         x[batch_id_b, pos_s:pos_e] = backup

for n in tqdm(range(conf.epoch)):
    for _ in range(1 + datar.totalBinSize // conf.batchSize):
        # get data
        slice = datar.makeBatch(conf.batchSize).to(trainingDevice)
        # glicher(slice)
        source = slice[:, :conf.contextSize]
        target = slice[:, conf.contextSize]
        # forward
        out = infer_pipeline(source)
        # calc loss
        loss = lossfunc(out, target)
        # backward and optimize
        optim.zero_grad()
        loss.backward()
        optim.step()
    print(f'Epoch {n} Loss: {loss.item()}')
    if n % 128 == 127:
        state_dict = langModel.state_dict()
        torch.save(state_dict, 'lang_model.pth')
        print('Model saved')

test(source[0:1])