import copy
import datetime
import pathlib
import random
import subprocess

import fire
import numpy as np
import sklearn.model_selection
import torch
import tqdm

import emojicompletion


def main(
    batch_size=10,
    train_ratio=0.8,
    lr=1e-4,
    weight_decay=1e-5,
    num_epoch=20,
):
    tokenizer = emojicompletion.data.Tokenizer()
    model = emojicompletion.model.Transformer(
        n_token=len(tokenizer.dictionary))
    candidates, case_dict = emojicompletion.data.SlackEmojiCompletionDataset(
        directory=pathlib.Path(__file__).parent /
        'data').load(filter_by_vocabulary=True)
    candidates = list(candidates)
    keys_train, keys_test = sklearn.model_selection.train_test_split(
        list(case_dict.keys()),
        test_size=1 - train_ratio,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    git_commit_id = (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )
    results_dir = (
        pathlib.Path(__file__).parent
        / "results"
        / f'batch{batch_size}_lr{lr}_commit{git_commit_id}_{datetime.datetime.now().strftime("%Y%m%d%H%M")}'
    )
    results_dir.mkdir(exist_ok=True, parents=True)

    def data_generator(keys, desc=None):
        for key in tqdm.tqdm(keys, desc=desc):
            if case_dict[key]:
                target = random.choice(case_dict[key])
                yield key + '/' + target, True
            target = random.choice(candidates)
            yield key + target, False

    for epoch in range(num_epoch):
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()
            batch = []
            running_loss = 0.0
            running_n = 0
            confmat = np.zeros((2, 2), dtype=int)
            model.train()
            for i, (x, y) in enumerate(
                    data_generator(keys_train, desc=f'[train, epoch{epoch}]')):
                batch.append((x, y))
                if len(batch) >= batch_size:
                    xs = [item[0] for item in batch]
                    maxlen = max(len(x) for x in xs)

                    xs = torch.stack(
                        [
                            tokenizer.tokenize(x + "*" * (maxlen - len(x)))
                            for x in xs
                        ],
                        dim=-1,
                    )
                    ys = torch.stack(
                        [torch.FloatTensor([item[1]]) for item in batch])

                    predict = model(xs)[0]
                    loss = torch.nn.BCEWithLogitsLoss()(predict, ys)

                    if phase == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    ys_pred = torch.nn.Sigmoid()(predict) > 0.5
                    for y, y_pred in zip(ys, ys_pred):
                        confmat[int(y), int(y_pred)] += 1
                    running_n += xs.size(0)
                    running_loss += loss.item() * xs.size(0)
                    batch = []
                if phase == "train" and i % 1000 == 0 and running_n > 0:
                    print(running_loss / running_n)
                    print(confmat)
            print(running_loss / running_n)
            print(confmat)
        torch.save(
            copy.deepcopy(model).to("cpu").state_dict(),
            results_dir / f"model_epoch{epoch}.pth",
        )

if __name__ == '__main__':
    fire.Fire(main)
