import argparse
import os

from conllu import parse
import nltk
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class UDDatasetBuilder:
    def __init__(self):
        self.train_data = []
        self.dev_data = []
        self.test_data = []
        self.train_dataset = []
        self.dev_dataset = []
        self.test_dataset = []
        self.pos_tags = []
        self.word2idx = {}
        self.idx2word = {}
        self.pos2idx = {}
        self.idx2pos = {}
        self.load_data()
        self.build_vocab()
        self.build_pos_vocab()
        self.build_datasets()

    def load_data(self):
        DATA_DIR = './data/ud-treebanks-v2.11/UD_English-Atis'
        with open(os.path.join(DATA_DIR, 'en_atis-ud-train.conllu'), 'r', encoding='utf-8') as f:
            self.train_data = parse(f.read())
        with open(os.path.join(DATA_DIR, 'en_atis-ud-dev.conllu'), 'r', encoding='utf-8') as f:
            self.dev_data = parse(f.read())
        with open(os.path.join(DATA_DIR, 'en_atis-ud-test.conllu'), 'r', encoding='utf-8') as f:
            self.test_data = parse(f.read())

    def build_vocab(self):
        self.word2idx['PAD'] = 0
        self.idx2word[0] = 'PAD'
        for sentence in self.train_data:
            for token in sentence:
                word = token['form']
                if word not in self.word2idx:
                    self.word2idx[word] = len(self.word2idx)
                    self.idx2word[len(self.idx2word)] = word
        self.word2idx['UNK'] = len(self.word2idx)
        self.idx2word[len(self.idx2word)] = 'UNK'

    def get_word_idx(self, word):
        return self.word2idx.get(word, self.word2idx['UNK'])

    def build_pos_vocab(self):
        self.pos2idx['PAD'] = 0
        self.idx2pos[0] = 'PAD'
        for sentence in self.train_data:
            for token in sentence:
                pos = token['upos']
                if pos not in self.pos2idx:
                    self.pos2idx[pos] = len(self.pos2idx)
                    self.idx2pos[len(self.idx2pos)] = pos
        self.pos2idx['UNK'] = len(self.pos2idx)
        self.idx2pos[len(self.idx2pos)] = 'UNK'
        self.pos_tags = list(self.pos2idx.keys())

    def get_pos_idx(self, pos):
        return self.pos2idx.get(pos, self.pos2idx['UNK'])

    def build_datasets(self):
        for sentence in self.train_data:
            self.train_dataset.append({
                'form': torch.tensor([self.get_word_idx(token['form']) for token in sentence]),
                'tags': torch.tensor([self.get_pos_idx(token['upos']) for token in sentence])
            })
        for sentence in self.dev_data:
            self.dev_dataset.append({
                'form': torch.tensor([self.get_word_idx(token['form']) for token in sentence]),
                'tags': torch.tensor([self.get_pos_idx(token['upos']) for token in sentence])
            })
        for sentence in self.test_data:
            self.test_dataset.append({
                'form': torch.tensor([self.get_word_idx(token['form']) for token in sentence]),
                'tags': torch.tensor([self.get_pos_idx(token['upos']) for token in sentence])
            })

    def vocab_size(self):
        return len(self.word2idx)

    def pos_size(self):
        return len(self.pos2idx)


class UDDataset(Dataset):
    def __init__(self, builder: UDDatasetBuilder, type='train'):
        self.type = type
        self.builder = builder
        if type == 'train':
            self.dataset = self.builder.train_dataset
        elif type == 'dev':
            self.dataset = self.builder.dev_dataset
        elif type == 'test':
            self.dataset = self.builder.test_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class BiLSTM(nn.Module):
    def __init__(self, n_layers, word_embedding_dim,  hidden_dim, vocab_size, pos_size, dropout=0.5):
        super(BiLSTM, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size, word_embedding_dim)
        # (N, L, E) -> (N, L, 2 * H)
        self.lstm = nn.LSTM(word_embedding_dim, hidden_dim,
                            num_layers=n_layers, bidirectional=True,
                            dropout=dropout, batch_first=True)
        # *2 due to bidirectional
        # (N, L, 2 * H) -> (N, L, P)
        self.fc = nn.Linear(hidden_dim * 2, pos_size)
        # (N, L, P) -> (N, L, P)

    def forward(self, x):
        x = self.word_embedding(x)
        x, _ = self.lstm(x)
        x = self.fc(x)
        x = F.softmax(x, dim=-1)
        return x


def pad_collate(batch):
    # batch.sort(key=lambda x: len(x['form']), reverse=True)
    # print(batch)
    form = [x['form'] for x in batch]
    tags = [x['tags'] for x in batch]
    form = nn.utils.rnn.pad_sequence(form, batch_first=True, )
    tags = nn.utils.rnn.pad_sequence(tags, batch_first=True)
    return form, tags


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('using device:', device)
    dataset = UDDatasetBuilder()

    train_dataset = UDDataset(dataset, type='train')
    dev_dataset = UDDataset(dataset, type='dev')

    print('train dataset size:', len(train_dataset))
    print('dev dataset size:', len(dev_dataset))
    print('pos tags:', dataset.pos_tags)

    INPUT_DIM = dataset.vocab_size()
    WORD_EMBEDDING_DIM = 100
    OUTPUT_DIM = dataset.pos_size()
    HIDDEN_DIM = 128
    DROPOUT = 0.5
    NUM_LAYERS = 2

    BATCH_SIZE = 32
    NUM_EPOCHS = 30
    LEARNING_RATE = 0.01

    model = BiLSTM(NUM_LAYERS, WORD_EMBEDDING_DIM, HIDDEN_DIM,
                   INPUT_DIM, OUTPUT_DIM, DROPOUT).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_data_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, collate_fn=pad_collate)
    dev_data_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE,
                                 shuffle=True, num_workers=8, collate_fn=pad_collate)

    # traning loop
    tr_losses = []
    dev_losses = []
    accuracy = []
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = -1
        for batch in train_data_loader:
            form, tags = batch
            form = form.to(device)
            tags = tags.to(device)
            optimizer.zero_grad()
            output = model(form)
            output = torch.flatten(output, end_dim=1)
            tags = torch.flatten(tags)
            # print(output.size(), tags.size())
            loss = criterion(output, tags)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        # compute dev loss
        dev_loss = 0
        acc = 0
        model.eval()
        with torch.no_grad():
            for batch in dev_data_loader:
                form, tags = batch
                form = form.to(device)
                tags = tags.to(device)
                # print(form.size(), tags.size())
                # print(tags)
                output = model(form)
                # preds = torch.argmax(output, dim=2)
                # print(preds)
                output = torch.flatten(output, end_dim=1)
                preds = torch.argmax(output, dim=1).cpu().numpy()
                # exit()
                tags = torch.flatten(tags)
                loss = criterion(output, tags)
                dev_loss += loss.item()
                tags = tags.cpu().numpy()
                non_padding = (tags != 0)
                # print(preds.shape, tags.shape)
                preds = preds[non_padding]
                tags = tags[non_padding]
                acc += accuracy_score(tags, preds)
        accuracy.append(acc / len(dev_data_loader))
        loss = epoch_loss / len(train_data_loader)
        dloss = dev_loss / len(dev_data_loader)
        dev_losses.append(dloss)
        if dloss <= min(dev_losses):
            torch.save(model.state_dict(), 'model.pt')
        tr_losses.append(loss)

        print(
            f'Epoch: {epoch + 1:02} | Loss: {loss:.5f} | Dev Loss: {dloss:.5f}')

    return tr_losses, dev_losses, accuracy


def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('using device:', device)
    dataset = UDDatasetBuilder()
    test_dataset = UDDataset(dataset, type='test')
    print('test dataset size:', len(test_dataset))
    INPUT_DIM = dataset.vocab_size()
    WORD_EMBEDDING_DIM = 100
    OUTPUT_DIM = dataset.pos_size()
    HIDDEN_DIM = 128
    DROPOUT = 0.5
    NUM_LAYERS = 2

    BATCH_SIZE = 32
    test_data_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, collate_fn=pad_collate)
    model = BiLSTM(NUM_LAYERS, WORD_EMBEDDING_DIM, HIDDEN_DIM,
                   INPUT_DIM, OUTPUT_DIM, DROPOUT).to(device)
    if not os.path.exists('model.pt'):
        raise Exception('model.pt not found')
    model.load_state_dict(torch.load('model.pt'))
    model.eval()
    with torch.no_grad():
        all_preds = np.array([])
        all_tags = np.array([])
        for batch in test_data_loader:
            form, tags = batch
            form = form.to(device)
            output = model(form)
            output = torch.flatten(output, end_dim=1)
            preds = torch.argmax(output, dim=1).cpu().numpy()
            tags = torch.flatten(tags).numpy()
            non_padding = (tags != 0)
            preds = preds[non_padding]
            tags = tags[non_padding]
            # print(preds, tags)
            all_preds = np.concatenate((all_preds, preds))
            all_tags = np.concatenate((all_tags, tags))
        print(classification_report(all_tags, all_preds,
                                    labels=list(range(1, 14)), zero_division=0))
    return all_preds, all_tags


def test_one():
    sentence = input()
    tokens = nltk.word_tokenize(sentence)
    tokens = [token.lower() for token in tokens]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = UDDatasetBuilder()
    form = torch.tensor([dataset.word2idx.get(token, 0)
                        for token in tokens]).to(device)
    INPUT_DIM = dataset.vocab_size()
    WORD_EMBEDDING_DIM = 100
    OUTPUT_DIM = dataset.pos_size()
    HIDDEN_DIM = 128
    DROPOUT = 0.5
    NUM_LAYERS = 2

    model = BiLSTM(NUM_LAYERS, WORD_EMBEDDING_DIM, HIDDEN_DIM,
                   INPUT_DIM, OUTPUT_DIM, DROPOUT).to(device)
    if not os.path.exists('model.pt'):
        raise Exception('model.pt not found')
    model.load_state_dict(torch.load('model.pt'))
    model.eval()
    with torch.no_grad():
        output = model(form)
        preds = torch.argmax(output, dim=1).cpu().numpy()
        for token, tag in zip(tokens, preds):
            print(f"{token}\t{dataset.idx2pos[tag]}")


if __name__ == '__main__':
    # parse the arguments
    parser = argparse.ArgumentParser(
        prog='POS tagger',
        description='Generates POS tags for sentences using a BiLSTM Model',
        epilog='author: Kannav Mehta')
    parser.add_argument('--train')
    parser.add_argument('--test')
    args = parser.parse_args()

    if args.train or args.test:
        if args.train:
            train()
        if args.test:
            test()
    else:
        test_one()
