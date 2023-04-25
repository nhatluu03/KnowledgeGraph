from underthesea import pos_tag
from sklearn.metrics import f1_score
import os


# Define your own function to map the tags to your desired output format
def map_tags(tag):
    if tag.startswith("Nc"):
        return "N" # map common nouns to "Nc"
    elif tag.startswith("V"):
        return "V"  # map verbs to "V"
    elif tag == "A":
        return "A" # map adjectives to "Adj"
    elif tag == "C":
        return "C"   # map coordinating conjunctions to "C"
    elif tag == "CH":
        return "CH" # map punctuation marks to "Punct"
    else:
        return tag   # keep the tag as is for all other cases

test_file = os.path.abspath('pos_test.txt')
valid_file = os.path.abspath('pos_valid.txt')
parser = VnCoreNLP("VnCoreNLP/VnCoreNLP-1.1.1.jar", annotators="parse", max_heap_size='-Xmx500m')    

y_true = []
y_pred = []

with open(test_file, 'r', encoding='utf-8') as test_f, open(valid_file, 'r', encoding='utf-8') as valid_f:
    for test_line, valid_line in zip(test_f, valid_f):
        
        test_tags = [tag for word, tag in parser.pos_tag(test_line.strip())]
        valid_tags = valid_line.strip().split(' ')
        
        # Map the tags to your desired output format
        test_tags = [map_tags(tag) for tag in test_tags]

        print(f"Sentence: " + test_line)
        print(f"Test tags: ", test_tags)
        print(f"Valid tags: ", valid_tags)
        print()
        
        

        # Pad shorter list with empty strings to match length of longer list (Padding)
        if len(test_tags) > len(valid_tags):
            valid_tags += [''] * (len(test_tags) - len(valid_tags))
        else:
            test_tags += [''] * (len(valid_tags) - len(test_tags))

        y_true.extend(valid_tags)
        y_pred.extend(test_tags)

f1 = f1_score(y_true, y_pred, average='weighted')

print(f'F1-score: {f1:.4f}')  // 0.8273
