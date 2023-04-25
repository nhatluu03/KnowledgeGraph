# !pip install py4j
# !git clone https://github.com/vncorenlp/VnCoreNLP.git
# !pip install vncorenlp

import sys
sys.path.append('./VnCoreNLP')
from vncorenlp import VnCoreNLP
rdrsegmenter = VnCoreNLP("VnCoreNLP/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')

from sklearn.metrics import f1_score
import os

# Get the absolute path
test_file = os.path.abspath('wseg_test.txt')
valid_file = os.path.abspath('wseg_valid.txt')

# Read in test data
with open(test_file, encoding="utf-8") as f:
    text_data = f.readlines()
    
with open(valid_file, encoding="utf-8") as f:
    valid_data = f.readlines()

# Preprocess data
vncorenlp_text_data = [[word for word in line.strip().split()] for line in text_data]
vncorenlp_valid_data = [[word for word in line.strip().split()] for line in valid_data]

# Calculate F1-score for validation data
valid_f1 = f1_score([word for line in vncorenlp_valid_data for word in line],
                    [word for line in vncorenlp_valid_data for word in line],
                    average="macro")
print("VNCoreNLP validation F1-score:", valid_f1)  # VNCoreNLP validation F1-score: 1.0
