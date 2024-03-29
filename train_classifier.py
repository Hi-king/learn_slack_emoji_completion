from cgitb import enable
import copy
import datetime
import inspect
import json
import pathlib
import random
import subprocess
from typing import Optional

import fire
import more_itertools
import numpy as np
import pandas as pd
import sklearn.model_selection
import torch
import wandb
import tqdm

import emojicompletion


class RandomAccessQueue:
    def __init__(self, maxsize) -> None:
        self.list = []
        self.maxsize = maxsize
    
    def put(self, item) -> None:
        self.list.append(item)
        if len(self.list) > self.maxsize:
            self.list.pop(0) # slow...

def main(
    batch_size=10,
    train_ratio=0.8,
    lr=1e-4,
    weight_decay=1e-5,
    num_epoch=20,
    save_interval_epoch=100,
    validation_interval_epoch=10,
    enable_hard_negative=False,
    enable_adaptive_hard_negative=False,
    enable_positional_encoding=True,
    enable_wandb=False,
    seed=42,
    dropout=0.1,
    from_model=None,
    model_type='transformer',
    name_prefix='',
    num_layers=3,
    output_type='bi',
    should_use_gpu=False,
    **kwargs,
):
    assert not kwargs, kwargs # check undefined cmdline args
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    use_gpu = torch.cuda.is_available()
    if should_use_gpu: assert use_gpu
    git_commit_id = (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )
    results_dir = (
        pathlib.Path(__file__).parent
        / "results"
        / f'{name_prefix}_classify_{model_type}_batch{batch_size}_lr{lr}_commit{git_commit_id}_hardneg{enable_hard_negative}_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}'
    )
    results_dir.mkdir(exist_ok=True, parents=True)

    candidates, case_dict = emojicompletion.data.SlackEmojiCompletionDataset(
        directory=pathlib.Path(__file__).parent /
        'data').load(filter_by_vocabulary=True)
    candidates = list(candidates)
    candidate2key = {
        value: i for i, value in enumerate(candidates)
    }

    tokenizer = emojicompletion.data.Tokenizer()
    if model_type == 'transformer':
        model = emojicompletion.model.Transformer(
            dropout=dropout,
            n_token=len(tokenizer.dictionary),
            num_layers=num_layers,
            positional_encoding=enable_positional_encoding,
            output_type=output_type,
            output_dim=len(candidates)
        )
    elif model_type == 'lstm':
        model = emojicompletion.model.SimpleLSTM(
            dropout=dropout,
            num_layers=num_layers,
            n_token=len(tokenizer.dictionary),
            output_type=output_type,
            output_dim=len(candidates)
        )

    if from_model:
        model.load_state_dict(torch.load(from_model))
    criterion = torch.nn.BCEWithLogitsLoss()
    # criterion = torch.nn.CrossEntropyLoss()
    if use_gpu:
        model = model.cuda()
        criterion = criterion.cuda()
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
        for key in (inspect.getfullargspec(main).args + [
            "git_commit_id", "use_gpu", 'keys_train', 'keys_test', 'candidates'])
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


    def data_generator(keys, optional_queue: Optional[RandomAccessQueue], desc=None):
        for key in tqdm.tqdm(keys, desc=desc):
            # positive sample
            if case_dict[key]:
                target_vector = np.zeros(len(candidates), dtype=np.int64)
                for each_target in case_dict[key]:
                    target_vector[candidate2key[each_target]] = 1
                yield key, target_vector


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
            for i, (x, y) in enumerate(
                    data_generator(keys, optional_queue=None, desc=f'[{phase}, epoch{epoch}]')):
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
                    # ys = torch.stack(
                    #     [torch.IntTensor([candidate2key[item[1]] for item in batch])]
                    # )
                    # ys = torch.LongTensor([candidate2key[item[1]] for item in batch])
                    ys = torch.stack(
                        [torch.FloatTensor(item[1]) for item in batch])
                    if use_gpu:
                        xs = xs.cuda()
                        ys = ys.cuda()

                    predict = model(xs)
                    loss = criterion(predict, ys)

                    if phase == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    ys_pred: torch.Tensor = torch.nn.Sigmoid()(predict) > 0.5                    
                    # y_preds += ys_pred.T.to('cpu').numpy().astype(int)[0].tolist()
                    # y_trues += ys.T.to('cpu').numpy().astype(int)[0].tolist()
                    y_preds += torch.flatten(ys_pred).to('cpu').numpy().astype(int).tolist()
                    y_trues += torch.flatten(ys).to('cpu').numpy().astype(int).tolist()
                    # print(torch.flatten(ys_pred))
                    # print(y_preds)
                    # print(y_trues)
                    # for y, y_pred in zip(torch.flatten(ys), torch.flatten(ys_pred)):
                    #     # print(y, y_pred)
                    #     confmat[int(y), int(y_pred)] += 1
                    running_n += xs.size(0)
                    running_loss += loss.item() * xs.size(0)
                    batch = []
                # if phase == "train" and i % 3000 == 0 and running_n > 0:
                #     print(running_loss / running_n)
                #     print(confmat)
            print(running_loss / running_n)
            if epoch % validation_interval_epoch == 0:
                confmat = np.zeros((2, 2), dtype=int)
                for y, y_pred in zip(y_preds, y_trues):
                    # print(y, y_pred)
                    confmat[int(y), int(y_pred)] += 1
                print(confmat)
            # print(confmat)
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
                # for xs_str in more_itertools.chunked(candidates, n=batch_size):
                with torch.inference_mode():
                    xs = torch.stack(
                        [
                            tokenizer.tokenize(key)
                        ],
                        dim=-1,
                    )
                    if use_gpu:
                        xs = xs.cuda()
                    pred = torch.nn.Sigmoid()(model(xs)).to('cpu').numpy()
                    for i in range(len(pred[0])):
                        score = pred[0, i]
                        result.append(dict(candidate=candidates[i], score=score))
                df = pd.DataFrame(result).assign(key=key)
                df = df.sort_values(by="score", ascending=False).assign(rank=range(1, len(df)+1))

                first_hit = df[df["candidate"].isin(case_dict[key])].sort_values('rank').iloc[[0]]
                dfs.append(first_hit)
                print(first_hit)
            if enable_wandb:
                result_table = pd.concat(dfs)
                wandb.log(dict(
                    epoch=epoch,
                    table=wandb.Table(dataframe=result_table),
                    sum_rank=result_table["rank"].sum(),
                ) | {f'rank_{row["key"]}': row["rank"] for _,row in result_table.iterrows()})



if __name__ == '__main__':
    fire.Fire(main)
