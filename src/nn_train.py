import numpy as np
import pandas as pd

import torch
from transformers import *


BERT_MODEL_PATH = '/Users/zx/Documents/pretrained_model/bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(
    BERT_MODEL_PATH)
text = "Who was Jim Henson ?"
tokenized_text = tokenizer.tokenize(text)
print(tokenized_text)

# indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
# print(indexed_tokens)
indexed_tokens = tokenizer.encode(text, add_special_tokens=True)
print(indexed_tokens)
print(tokenizer.decode(indexed_tokens))

segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])


model = BertModel.from_pretrained(
    BERT_MODEL_PATH)
model.eval()

with torch.no_grad():
    print(tokens_tensor)
    last_hidden_states = model(tokens_tensor)
    print(len(last_hidden_states))
    last_hidden_states = last_hidden_states[0]
    print(last_hidden_states.shape)
#
# # tokens_tensor = tokens_tensor.to('cuda')
# # segments_tensors = segments_tensors.to('cuda')
# # model.to('cuda')
#
# # Predict hidden states features for each layer
# with torch.no_grad():
#     encoded_layers, _ = model(tokens_tensor, segments_tensors)
# # We have a hidden states for each of the 12 layers in model bert-base-uncased
# assert len(encoded_layers) == 12
#
# print([e.size() for e in encoded_layers])