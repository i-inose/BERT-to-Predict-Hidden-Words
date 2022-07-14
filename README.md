# Repository Description
This program uses BERT to predict hidden words.

# Program Description
Assign a sentence to the text variable.<br>
```
text = "you are very strong in japan."
```

Add a token [CLS] at the beginning of the text variable and a token [SEP] at the end.<br>
```
tokenized_text.insert(0, "[CLS]")
tokenized_text.append("[SEP]")
```

Replace the word you want to hide with a [MASK] token.<br>
```
tokenized_text[4] = "[MASK]"
```

Convert text to id.<br>
```
tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
tokens_tensor = torch.tensor([tokens])
```

The number of words to be hidden and the number of predicted results are written as arguments, and the results are displayed.<br>
```
_, predict_indexes = torch.topk(predictions[0, 4], k=5)
predict_tokens = tokenizer.convert_ids_to_tokens(predict_indexes.tolist())
print(predict_tokens)
```

# Result
![image](https://user-images.githubusercontent.com/78309273/178986669-2b86a293-1ae2-453b-914f-93183c010862.png)