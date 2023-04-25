!pip install underthesea
from underthesea import word_tokenize
from sklearn.metrics import f1_score
import os

# Get the absolute path
test_file = os.path.abspath('wseg_test.txt')
valid_file = os.path.abspath('wseg_valid.txt')

y_true = []
y_pred = []

with open(test_file, 'r', encoding='utf-8') as test_f, open(valid_file, 'r', encoding='utf-8') as valid_f:
    for test_line, valid_line in zip(test_f, valid_f):
        test_words = word_tokenize(test_line.strip())
        valid_words = valid_line.strip().split(' ')
        test_words = [token.replace(" ", "_") for token in test_words]

        # Pad shorter list with empty strings to match length of longer list
        if len(test_words) > len(valid_words):
            valid_words += [''] * (len(test_words) - len(valid_words))
        else:
            test_words += [''] * (len(valid_words) - len(test_words))

        y_true.extend(valid_words)
        y_pred.extend(test_words)

f1_metrics = f1_score(y_true, y_pred, average='weighted')
f1_score = f1_score(y_true, y_pred, average='weighted')
print(f'F1-score: {f1_score:.4f}')   // 0.5760
