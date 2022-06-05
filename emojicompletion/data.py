import json
import pathlib
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
    vocabulary = string.ascii_lowercase + string.digits + ' ' + '-' + '_' + '*' + '/'  # * as a padding, / as a separator

    def __init__(self) -> None:
        self.dictionary = Dictionary(self.vocabulary)

    def tokenize(self, sentence: str) -> torch.Tensor:
        return torch.tensor([
            self.dictionary.word2idx[token] for token in sentence
        ]).type(torch.int64)


class SlackEmojiCompletionDataset:

    def __init__(self, directory) -> None:
        self.directory = pathlib.Path(directory)
        self.candidate_characters = string.ascii_lowercase + string.digits + ' '

    def _is_in_vocabulary(self, sentence):
        return all((x in Tokenizer.vocabulary) for x in sentence)

    def load(self, filter_by_vocabulary: bool):
        candidates = {
            line.rstrip()[1:-1]
            for line in (self.directory / f'candidates.txt').open()
        }
        if filter_by_vocabulary:
            candidates = {
                candidate
                for candidate in candidates
                if self._is_in_vocabulary(candidate)
            }

        case_dict = {}
        for character in self.candidate_characters:
            for line in (self.directory / f'{character}.txt').open():
                datum = json.loads(line)
                case_dict[datum['key']] = [
                    candidate[1:-1] for candidate in datum['result']
                ]
                if filter_by_vocabulary:
                    case_dict[datum['key']] = [
                        candidate for candidate in case_dict[datum['key']]
                        if self._is_in_vocabulary(candidate)
                    ]
        return candidates, case_dict
