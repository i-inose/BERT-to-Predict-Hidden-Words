import torch
from transformers import BertTokenizer, BertForMaskedLM
import numpy as np

text = "you are very strong in japan."

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenized_text = tokenizer.tokenize(text)
tokenized_text.insert(0, "[CLS]")
tokenized_text.append("[SEP]")
# ['[CLS]', 'you', 'are', 'very', 'strong', 'in', 'japan', '.', '[SEP]

#strongを隠す
tokenized_text[4] = '[MASK]'

#テキストをidに変換する。
tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
tokens_tensor = torch.tensor([tokens])

model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()

with torch.no_grad():
    outputs = model(tokens_tensor)
    predictions = outputs[0]

#予測結果を5つ出す。
_, predict_indexes = torch.topk(predictions[0, 4], k=5)
predict_tokens = tokenizer.convert_ids_to_tokens(predict_indexes.tolist())
print(predict_tokens)