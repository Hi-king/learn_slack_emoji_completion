import pathlib

import fire

import emojicompletion


def main():
    candidates, case_dict = emojicompletion.data.SlackEmojiCompletionDataset(directory=pathlib.Path(__file__).parent / 'data').load()

if __name__ == '__main__':
    fire.Fire(main)
