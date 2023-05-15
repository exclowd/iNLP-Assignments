import argparse
import os
import pickle
import shutil
from typing import Dict, Optional

import datasets
import nltk
import torch
from datasets import load_dataset
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset



def train(dataset: str):
    data_module = SSTDataModule(preprocess=False)
    data_module.prepare_data()
    data_module.setup()
    train_data = data_module.train_dataloader()
    for batch in train_data:
        print(batch)
        break
    val_data = data_module.val_dataloader()
    for batch in val_data:
        print(batch)
        break


def test(dataset: str):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        'elmo.py',
        'python3 elmo.py',
        'train elmo embeddings'
    )

    parser.add_argument('dataset', choices=['sst', 'multi_nli'])
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    if args.train:
        train(args.dataset)
    if args.test:
        test(args.dataset)
