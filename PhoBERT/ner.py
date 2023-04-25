# !git clone --single-branch --branch fast_tokenizers_BARTpho_PhoBERT_BERTweet https://github.com/datquocnguyen/transformers.git
# !cd transformers
# !pip3 install -e .

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Load PhoBERT tokenizer and model for token classification
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
model = AutoModelForTokenClassification.from_pretrained("vinai/phobert-base", num_labels=4)

# Define the labels for NER
labels = ['B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'O']

# Define a Vietnamese text to analyze
text = "Công ty cổ phần Sài Gòn là một trong những công ty hàng đầu tại Việt Nam."

# Tokenize the text and convert to input IDs
tokens = tokenizer.encode(text, add_special_tokens=True)
input_ids = torch.tensor([tokens])

# Feed the input IDs into the model to obtain predicted token labels
outputs = model(input_ids)
predicted_labels = torch.argmax(outputs.logits, dim=2)

# Map the predicted token labels back to their corresponding NER labels
predicted_ner = [labels[i] for i in predicted_labels[0].tolist()]

# Print out the predicted NER labels for each token in the text
for i, token in enumerate(tokenizer.tokenize(text)):
    print(token + " : " + predicted_ner[i])
