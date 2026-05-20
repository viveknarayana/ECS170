'''
Concrete IO class for stage 4 IMDb text classification.
'''

from local_code.base_class.dataset import dataset
import os
import re
from collections import Counter

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# following the guidelines given in the dataset: https://machinelearningmastery.com/clean-text-machine-learning-python/ 
# this is for reading!!!
for resource in ('corpora/stopwords', 'tokenizers/punkt', 'tokenizers/punkt_tab'):
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource.split('/')[-1], quiet=True)

STOPWORDS = set(stopwords.words('english'))
HTML_TAG_RE = re.compile(r'<[^>]+>')

PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'


def clean_and_tokenize(text):
    text = HTML_TAG_RE.sub(' ', text).lower()
    tokens = word_tokenize(text)
    return [t for t in tokens if t.isalpha() and t not in STOPWORDS]


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    vocab_size = 10000
    sequence_length = 200

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
        self.vocab = None

    def _read_split(self, split_folder):
        X_tokens, y = [], []
        for label_name, label_id in (('neg', 0), ('pos', 1)):
            label_folder = os.path.join(split_folder, label_name)
            for fname in sorted(os.listdir(label_folder)):
                if not fname.endswith('.txt'):
                    continue
                with open(os.path.join(label_folder, fname), 'r', encoding='utf-8') as f:
                    X_tokens.append(clean_and_tokenize(f.read()))
                    y.append(label_id)
        return X_tokens, y

    def _build_vocab(self, train_tokens):
        counter = Counter()
        for tokens in train_tokens:
            counter.update(tokens)
        vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        for word, _ in counter.most_common(self.vocab_size - 2):
            vocab[word] = len(vocab)
        return vocab

    def _encode(self, tokens):
        unk = self.vocab[UNK_TOKEN]
        ids = [self.vocab.get(t, unk) for t in tokens]
        # pad/truncate so every example is the same length and we can stack as one tensor
        if len(ids) >= self.sequence_length:
            return ids[:self.sequence_length]
        return ids + [self.vocab[PAD_TOKEN]] * (self.sequence_length - len(ids))

    def load_file(self, file_name=None):
        print('loading data...')
        root = self.dataset_source_folder_path
        train_tokens, train_y = self._read_split(os.path.join(root, 'train'))
        test_tokens, test_y = self._read_split(os.path.join(root, 'test'))

        self.vocab = self._build_vocab(train_tokens)

        train_X = [self._encode(t) for t in train_tokens]
        test_X = [self._encode(t) for t in test_tokens]

        return {
            'train': {'X': train_X, 'y': train_y},
            'test': {'X': test_X, 'y': test_y}
        }

    def load(self):
        return self.load_file()
