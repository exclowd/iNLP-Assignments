from typing import Dict, Optional

import os
import pickle
import shutil

from torch.utils.data import Dataset, DataLoader
from lightning import LightningDataModule
import datasets
from datasets import load_dataset
import nltk


class SSTDataset(Dataset):
    def __init__(self, data_dir: str = './data', split_name: str = 'train'):
        self.data_dir: str = data_dir
        self.split_name: str = split_name
        # load data from disk
        dataset_path = os.path.join(self.data_dir, f'sst_{self.split_name}.pkl')
        print(f'Loading {self.split_name} dataset from {dataset_path}')
        with open(dataset_path, 'rb') as f:
            self.dataset = pickle.load(f)
        with open(os.path.join(self.data_dir, 'sst_stoi.pkl'), 'rb') as f:
            stoi = pickle.load(f)
        itos = {v: k for k, v in stoi.items()}
        self.stoi = stoi
        self.itos = itos

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # create character ids
        
        return self.dataset[idx]


class SSTDataModule(LightningDataModule):
    def __init__(self,
                 data_dir: str = './data',
                 batch_size: int = 64,
                 num_workers: int = 4,
                 pin_memory: bool = False,
                 preprocess: bool = True
                 ):
        super().__init__()
        self.name = 'sst'
        self.data_dir: str = data_dir

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.preprocess = preprocess

        self.dataset: Optional[datasets.Dataset] = None
        self.data_train: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None

    def tokenize(self, dataset, stoi: Dict[str, int]):
        data = [[] for _ in range(len(dataset['sentence']))]
        for i, sent in enumerate(dataset['sentence']):
            tokens = [token.lower() for token in nltk.word_tokenize(sent)]
            data[i] = [2] + [stoi.get(token, 1) for token in tokens] + [3]
        return data

    def prepare_data(self) -> None:
        # tokenize and build vocab here then save to disk as pickle
        dataset = load_dataset('sst', name='default', split='train')
        assert isinstance(dataset, datasets.Dataset)
        stoi = {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3}
        itos = {0: '<pad>', 1: '<unk>', 2: '<bos>', 3: '<eos>'}
        data = [[] for _ in range(len(dataset['sentence']))]
        for i, sent in enumerate(dataset['sentence']):
            tokens = [token.lower() for token in nltk.word_tokenize(sent)]
            for token in tokens:
                if token not in stoi:
                    stoi[token] = len(stoi)
                    itos[len(itos)] = token
            data[i] = [2] + [stoi[token] for token in tokens] + [3]
        if self.preprocess:
            shutil.rmtree(self.data_dir)

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        # dump train data and vocab
        train_path = os.path.join(self.data_dir, 'sst_train.pkl')
        if not os.path.exists(train_path):
            with open(train_path, 'wb+') as f:
                pickle.dump(data, f)
        stoi_path = os.path.join(self.data_dir, 'sst_stoi.pkl')
        if not os.path.exists(stoi_path):
            with open(stoi_path, 'wb+') as f:
                pickle.dump(stoi, f)
        # dump test data
        test_data_path = os.path.join(self.data_dir, 'sst_test.pkl')
        if not os.path.exists(test_data_path):
            test_dataset = load_dataset('sst', name='default', split='test')
            test_data = self.tokenize(test_dataset, stoi)
            with open(test_data_path, 'wb+') as f:
                pickle.dump(test_data, f)
        # dump val data
        val_data_path = os.path.join(self.data_dir, 'sst_validation.pkl')
        if not os.path.exists(val_data_path):
            val_dataset = load_dataset('sst', name='default', split='validation')
            val_data = self.tokenize(val_dataset, stoi)
            with open(val_data_path, 'wb+') as f:
                pickle.dump(val_data, f)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'fit' or stage is None:
            self.data_train = SSTDataset(data_dir=self.data_dir, split_name='train')
            self.data_val = SSTDataset(data_dir=self.data_dir, split_name='validation')
        if stage == 'test':
            self.data_test = SSTDataset(data_dir=self.data_dir, split_name='test')

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn
        )

