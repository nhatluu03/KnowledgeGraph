!git clone --single-branch --branch fast_tokenizers_BARTpho_PhoBERT_BERTweet https://github.com/datquocnguyen/transformers.git
!cd transformers
!pip install -e .

import torch
from transformers import AutoModel, AutoTokenizer
from transformers import MBartForConditionalGeneration, MBart50Tokenizer

# Load the model and tokenizer
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")
tokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50")

# Define the input text
text = "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum."

# Encode the input text using the MBart50Tokenizer
input_ids = tokenizer(text, return_tensors='pt')['input_ids']

# Generate a summary using the MBartForConditionalGeneration model
summary_ids = model.generate(input_ids, max_length=50, num_beams=4, length_penalty=2.0)

# Decode the summary tokens back into text
summary_text = tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)

# Print the input text and summary
print("Input text:\n", text)
print("Summary text:\n", summary_text)
