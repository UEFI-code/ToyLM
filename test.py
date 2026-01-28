import EnDecoder
import LangModel
import torch
import dataset as dataset
import gpu_chooser

contextSize = 64

datar = dataset.DataWarpper(contextSize, './demo_txt_dataset')

device = gpu_chooser.choose_gpu()

# Load Encoder-Decoder model
en_decoder = EnDecoder.EnDecoder(embeddingDim=4)
en_decoder.load_state_dict(torch.load('encoder_decoder.pth'))
en_decoder = en_decoder.to(device)

# Load Language Model
langModel = LangModel.LangModel(max_seq_len=contextSize, embedding_dim=4, hidden_dim=64)
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
        test_batch = torch.concat((test_batch[:, 1:], torch.tensor([[byte]], device=device, dtype=torch.long)), dim=1)
    print(f'Final Result: {res}')

source, _ = datar.makeBatch(1)
source = source.to(device)
test(source)