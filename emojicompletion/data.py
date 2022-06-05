import string

import torch


class Dictionary(object):
    """https://github.com/pytorch/examples/blob/2bf23f105237e03ee2501f29670fb6a9ca915096/word_language_model/data.py#L5"""
    def __init__(self, vocabulary=[]):
        self.word2idx = {}
        self.idx2word = []
        for token in vocabulary:
            self.add_word(token)

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

class Tokenizer:
    def __init__(self) -> None:
        vocabulary = string.ascii_lowercase + string.digits + ' '
        self.dictionary = Dictionary(vocabulary)
    
    def tokenize(self, sentence: str) -> torch.Tensor:
        return torch.tensor([self.dictionary.word2idx[token] for token in sentence]).type(torch.int64)
