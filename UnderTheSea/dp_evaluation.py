# !pip install underthesea[deep]
from underthesea import dependency_parse
from underthesea import word_tokenize
from sklearn.metrics import f1_score
import os

test_file = os.path.abspath('dp_test.txt')
valid_file = os.path.abspath('dp_valid.txt')

y_true = []
y_pred = []


with open(test_file, 'r', encoding='utf-8') as test_f, open(valid_file, 'r', encoding='utf-8') as valid_f:
    for test_line, valid_line in zip(test_f, valid_f):
        print(test_line)
        test_line = [word.replace(" ", "_") for word in word_tokenize(test_line)]
        test_line = ' '.join(test_line)
        print(test_line)
        
        test_labels = [label[2] for label in dependency_parse(test_line.strip())]
        valid_labels = valid_line.strip().split(' ')
        print(test_labels)
        print(valid_labels)
        print()

        # Pad shorter list with empty strings to match length of longer list
        if len(test_labels) > len(valid_labels):
            valid_labels += [''] * (len(test_labels) - len(valid_labels))
        else:
            test_labels += [''] * (len(valid_labels) - len(test_labels))

        y_true.extend(valid_labels)
        y_pred.extend(test_labels)

f1 = f1_score(y_true, y_pred, average='weighted')

print(f'F1-score: {f1:.4f}')   # F1-score: 0.6623
