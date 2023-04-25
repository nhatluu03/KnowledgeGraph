# !pip install underthesea
from underthesea import ner
from sklearn.metrics import f1_score
import os

test_file = os.path.abspath('ner_test.txt')
valid_file = os.path.abspath('ner_valid.txt')

y_true = []
y_pred = []


with open(test_file, 'r', encoding='utf-8') as test_f, open(valid_file, 'r', encoding='utf-8') as valid_f:
    for test_line, valid_line in zip(test_f, valid_f):
        print(test_line)
        test_line = [word.replace(" ", "_") for word in word_tokenize(test_line)]
        test_line = ' '.join(test_line)
        print(test_line)
        
        test_labels = [label[3] for label in ner(test_line.strip())]
        valid_label = valid_line.strip().split(' ')
        
        # Map the tags to your desired output format
        test_tags = [map_tags(tag) for tag in test_tags]

        print(f"Tokenized labels: ", test_tags)
        print(f"Test labels: ", test_tags)
        print(f"Valid labels: ", valid_tags)
        print()

        # Pad shorter list with empty strings to match length of longer list
        if len(test_tags) > len(valid_tags):
            valid_tags += [''] * (len(test_tags) - len(valid_tags))
        else:
            test_tags += [''] * (len(valid_tags) - len(test_tags))

        y_true.extend(valid_tags)
        y_pred.extend(test_tags)

f1 = f1_score(y_true, y_pred, average='weighted')

print(f'F1-score: {f1:.4f}')   # F1-score: 0.7778
