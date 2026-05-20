'''
Concrete IO class for stage 4 joke generation.
'''

from local_code.base_class.dataset import dataset
import os
import csv
from collections import Counter

import nltk
from nltk.tokenize import word_tokenize


for resource in ('tokenizers/punkt', 'tokenizers/punkt_tab'):
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource.split('/')[-1], quiet=True)


PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
BOS_TOKEN = '<BOS>'
EOS_TOKEN = '<EOS>'

PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3


def tokenize_joke(text):
    # keep punctuation: jokes rely on '?' '!' for the punchline rhythm
    return word_tokenize(text.lower())


class Dataset_Loader_Generation(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    # truncate at this many tokens (most jokes are < 30); +2 for BOS/EOS
    max_seq_length = 40

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
        self.vocab = None
        self.id_to_token = None

    def _read_jokes(self, csv_path):
        jokes = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                jokes.append(row['Joke'])
        return jokes

    def _build_vocab(self, tokenized_jokes):
        # specials reserved at fixed ids; everything else assigned by frequency order
        counter = Counter()
        for tokens in tokenized_jokes:
            counter.update(tokens)
        vocab = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID, BOS_TOKEN: BOS_ID, EOS_TOKEN: EOS_ID}
        for word, _ in counter.most_common():
            if word not in vocab:
                vocab[word] = len(vocab)
        return vocab

    def _encode(self, tokens):
        # wrap with BOS/EOS, truncate, then pad to a common length
        ids = [BOS_ID] + [self.vocab.get(t, UNK_ID) for t in tokens] + [EOS_ID]
        if len(ids) > self.max_seq_length:
            ids = ids[:self.max_seq_length]
            ids[-1] = EOS_ID
        ids = ids + [PAD_ID] * (self.max_seq_length - len(ids))
        return ids

    def load_file(self, file_name=None):
        print('loading jokes')
        csv_path = os.path.join(self.dataset_source_folder_path, 'data')
        raw = self._read_jokes(csv_path)
        tokenized = [tokenize_joke(j) for j in raw]
        self.vocab = self._build_vocab(tokenized)
        self.id_to_token = {i: w for w, i in self.vocab.items()}

        encoded = [self._encode(t) for t in tokenized]

        # generation has no train/test split in the assignment; we train on all jokes
        return {
            'train': {'X': encoded, 'y': encoded},
            'test':  {'X': encoded, 'y': encoded}
        }

    def load(self):
        return self.load_file()
