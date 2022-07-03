from cgitb import enable
import copy
import datetime
import inspect
import json
import pathlib
import random
import subprocess

import fire
import more_itertools
import numpy as np
import pandas as pd
import sklearn.model_selection
import torch
import wandb
import tqdm

import emojicompletion


def main(
    batch_size=10,
    train_ratio=0.8,
    lr=1e-4,
    weight_decay=1e-5,
    num_epoch=20,
    save_interval_epoch=100,
    validation_interval_epoch=10,
    enable_hard_negative=False,
    enable_positional_encoding=True,
    enable_wandb=False,
    seed=42,
    dropout=0.1,
    from_model=None,
    model_type='transformer',
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    use_gpu = torch.cuda.is_available()
    git_commit_id = (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )
    results_dir = (
        pathlib.Path(__file__).parent
        / "results"
        / f'{model_type}_batch{batch_size}_lr{lr}_commit{git_commit_id}_hardneg{enable_hard_negative}_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}'
    )
    results_dir.mkdir(exist_ok=True, parents=True)

    tokenizer = emojicompletion.data.Tokenizer()
    if model_type == 'transformer':
        model = emojicompletion.model.Transformer(
            dropout=dropout,
            n_token=len(tokenizer.dictionary),
            positional_encoding=enable_positional_encoding,
        )
    elif model_type == 'lstm':
        model = emojicompletion.model.SimpleLSTM(
            dropout=dropout,
            n_token=len(tokenizer.dictionary),
        )

    if from_model:
        model.load_state_dict(torch.load(from_model))
    criterion = torch.nn.BCEWithLogitsLoss()
    if use_gpu:
        model = model.cuda()
        criterion = criterion.cuda()
    candidates, case_dict = emojicompletion.data.SlackEmojiCompletionDataset(
        directory=pathlib.Path(__file__).parent /
        'data').load(filter_by_vocabulary=True)
    candidates = list(candidates)
    keys_train, keys_test = sklearn.model_selection.train_test_split(
        list(case_dict.keys()),
        test_size=1 - train_ratio,
        random_state=seed,        
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # save params
    all_params = locals()
    params = {
        key: all_params[key]
        for key in (inspect.getfullargspec(main).args + ["git_commit_id", "use_gpu", 'keys_train', 'keys_test'])
    }
    print(params)
    json.dump(params, (results_dir / "params.json").open("w+"), indent=2)
    if enable_wandb:
        wandb.init(
            project='learn_slack_emoji_completion',
            name=results_dir.name,
            config=params,
        )
        wandb.watch(model, criterion, log="all", log_freq=100)



    if enable_hard_negative:
        hard_candidates_dict = {}
        for key in tqdm.tqdm(case_dict.keys(), desc="generate hard negative"):
            keycharset = set(key)
            hard_candidates = {candidate for candidate in candidates if keycharset.issubset(set(candidate))} - set(case_dict[key])
            hard_candidates_dict[key] = list(hard_candidates)

    def data_generator(keys, desc=None):
        for key in tqdm.tqdm(keys, desc=desc):
            # positive sample
            if case_dict[key]:
                target = random.choice(case_dict[key])
                yield key + '/' + target, True

            # negative sample
            target = random.choice(list(set(candidates) - set(case_dict[key])))
            yield key + '/' + target, False

            if enable_hard_negative:
                if hard_candidates_dict[key]:
                    target = random.choice(hard_candidates_dict[key])
                    yield key + '/' + target, False

    for epoch in range(num_epoch):
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
                keys = random.sample(keys_train,len(keys_train))
            else:
                model.eval()
                keys = random.sample(keys_train,len(keys_test))
            batch = []
            running_loss = 0.0
            running_n = 0
            confmat = np.zeros((2, 2), dtype=int)
            y_preds = []
            y_trues = []
            model.train()
            for i, (x, y) in enumerate(
                    data_generator(keys, desc=f'[{phase}, epoch{epoch}]')):
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
                    if use_gpu:
                        xs = xs.cuda()
                        ys = ys.cuda()

                    predict = model(xs)[0]
                    loss = criterion(predict, ys)

                    if phase == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    ys_pred: torch.Tensor = torch.nn.Sigmoid()(predict) > 0.5
                    y_preds += ys_pred.T.to('cpu').numpy().astype(int)[0].tolist()
                    y_trues += ys.T.to('cpu').numpy().astype(int)[0].tolist()
                    for y, y_pred in zip(ys, ys_pred):
                        confmat[int(y), int(y_pred)] += 1
                    running_n += xs.size(0)
                    running_loss += loss.item() * xs.size(0)
                    batch = []
                # if phase == "train" and i % 3000 == 0 and running_n > 0:
                #     print(running_loss / running_n)
                #     print(confmat)
            print(running_loss / running_n)
            if enable_wandb:
                wandb.log(dict(
                    epoch=epoch,
                    confmat=wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=y_trues, 
                        preds=y_preds,
                    )
                ))
                wandb.log({"epoch": epoch, f"{phase}_loss": running_loss / running_n})
            print(confmat)
        if epoch % save_interval_epoch == 0:
            torch.save(
                copy.deepcopy(model).to("cpu").state_dict(),
                results_dir / f"model_epoch{epoch}.pth",
            )

        if epoch % validation_interval_epoch == 0:
            rank_eval_keys = []
            for key in keys_test:
                if case_dict[key]:
                    rank_eval_keys.append(key)
                if len(rank_eval_keys) >= 5:
                    break
            dfs = []
            for key in rank_eval_keys:
                result = []
                for xs_str in more_itertools.chunked(candidates, n=batch_size):
                    with torch.inference_mode():
                        maxlen = max(len(x) for x in xs_str)

                        xs = torch.stack(
                            [
                                tokenizer.tokenize(key + '/' + x + "*" * (maxlen - len(x)))
                                for x in xs_str
                            ],
                            dim=-1,
                        )
                        if use_gpu:
                            xs = xs.cuda()
                        pred = torch.nn.Sigmoid()(model(xs)[0]).to('cpu').numpy()
                        for i in range(len(pred)):
                            score = pred[i,0]
                            result.append(dict(candidate=xs_str[i], score=score))
                df = pd.DataFrame(result).assign(key=key)
                df = df.sort_values(by="score", ascending=False).assign(rank=range(1, len(df)+1))
                dfs.append(df[df["candidate"] == case_dict[key][0]])
                print(df[df["candidate"] == case_dict[key][0]])
            if enable_wandb:
                result_table = pd.concat(dfs)
                wandb.log(dict(
                    epoch=epoch,
                    table=wandb.Table(dataframe=result_table),
                    sum_rank=result_table["rank"].sum(),
                ) | {f'rank_{row["key"]}': row["rank"] for _,row in result_table.iterrows()})



if __name__ == '__main__':
    fire.Fire(main)
