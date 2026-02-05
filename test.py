import EnDecoder
import LangModel
import torch
import dataset as dataset
import gpu_chooser
import conf

datar = dataset.DataWarpper(conf.contextSize, './demo_txt_dataset')

device = gpu_chooser.choose_gpu()

# Load Encoder-Decoder model
en_decoder = EnDecoder.EnDecoder(embeddingDim=conf.embedding_dim)
en_decoder.load_state_dict(torch.load('encoder_decoder.pth'))
en_decoder = en_decoder.to(device)

# Load Language Model
langModel = LangModel.LangModel(max_seq_len=conf.contextSize, embedding_dim=conf.embedding_dim, hidden_dim=conf.hidden_dim, depth=conf.depth)
langModel.load_state_dict(torch.load('lang_model.pth'))
langModel = langModel.to(device)

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
        test_batch = torch.concat((test_batch[:, 1:], torch.tensor([[byte]], device=device, dtype=torch.int)), dim=1)
    print(f'Final Result: {res}')

source = datar.makeBatch(1)
source = source.to(device)
test(source)