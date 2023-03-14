import argparse
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from language_model import RegexTokenizer
import math


class TextDataset(Dataset):
    def __init__(self, corpus_path, vocab_path):
        self.tk = RegexTokenizer()
        filename = os.path.basename(corpus_path)
        with open(corpus_path, "r") as f:
            data = f.read()
            self.data = self.tk.tokenize(data, pad_sentences=True)
        # create vocab file
        words = {}
        for sent in self.data:
            for word in sent:
                words[word] = words.get(word, 0) + 1
        words_sorted = sorted(words.items(), key=lambda x: x[1], reverse=True)
        words_sorted = words_sorted[2:6000]
        words_sorted = [word[0] for word in words_sorted]
        self.word_list = ['<unk>', '<start>', '<end>', '<pad>'] + list(words_sorted)
        with open(vocab_path, "w") as f:
            for word in self.word_list:
                f.write(word + '\n')
        # load vocab file
        with open(vocab_path, "r") as f:
            self.word_list = f.read().splitlines()
        self.vocab_size = len(self.word_list)
        self.word_dict = {word: i for i, word in enumerate(self.word_list)}
        self.itos = {i: word for i, word in enumerate(self.word_list)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sentence = [self.word_dict.get(word, 0) for word in self.data[idx]]
        return torch.tensor(sentence)


class NeuralLanguageModel(nn.Module):
    def __init__(self, n_layers, vocab_size, word_embedding_dim, hidden_dim, dropout):
        super(NeuralLanguageModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.embedding = torch.nn.Embedding(vocab_size, word_embedding_dim)
        self.lstm = torch.nn.LSTM(word_embedding_dim, hidden_dim, num_layers=n_layers,
                                  dropout=dropout, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, vocab_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.05
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.drop(x)
        x = self.fc(torch.flatten(x, end_dim=1))
        return x


def pad_sentences(batch):
    inp = [sent[:-1] for sent in batch]
    out = [sent[1:] for sent in batch]
    inp = nn.utils.rnn.pad_sequence(inp, batch_first=True, padding_value=3)
    out = nn.utils.rnn.pad_sequence(out, batch_first=True, padding_value=3)
    return inp, out


def train(args):
    print(args.corpus_path, args.vocab_path)

    txt_dataset = TextDataset(args.corpus_path, args.vocab_path)

    EPOCHS = 10
    EMBEDDING_DIM = 600
    VOCAB_SIZE = txt_dataset.vocab_size
    HIDDEN_DIM = 600
    LEARNING_RATE = 0.001
    DROPOUT = 0
    NUM_LAYERS = 1
    CLIP = 0.25

    BATCH_SIZE =  16
    NUM_WORKERS = 4



    train_set_size = int(0.7 * len(txt_dataset))
    dev_set_size = int(0.15 * len(txt_dataset))
    test_set_size = len(txt_dataset) - train_set_size - dev_set_size
    train_set, dev_set, test_set = random_split(txt_dataset, [train_set_size, dev_set_size, test_set_size])

    train_dataloader = DataLoader(train_set,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=NUM_WORKERS,
                                  collate_fn=pad_sentences)
    dev_data_loader = DataLoader(dev_set,
                                 batch_size=BATCH_SIZE,
                                 shuffle=True,
                                 num_workers=NUM_WORKERS,
                                 collate_fn=pad_sentences)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Training on device:", device)
    criterion = nn.CrossEntropyLoss().to(device)
    print("Vocab size:", VOCAB_SIZE)
    print("Embedding dim:", EMBEDDING_DIM)
    model = NeuralLanguageModel(NUM_LAYERS, VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, DROPOUT).to(device)
    # print number of parameters
    print("Number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    losses = []
    dev_losses = []
    for epoch in range(EPOCHS):
        total_loss = 0
        model.train()
        for batch in train_dataloader:
            input, target = batch
            input = input.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(input)
            target = torch.flatten(target)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimizer.step()
            total_loss += loss.item()
        total_loss = total_loss / len(train_dataloader)
        losses.append(total_loss)
        dev_loss = 0
        model.eval()
        with torch.no_grad():
            for batch in dev_data_loader:
                input, target = batch
                input = input.to(device)
                target = target.to(device)
                output = model(input)
                target = torch.flatten(target)
                loss = criterion(output, target)
                dev_loss += loss.item()
        dev_loss = dev_loss / len(dev_data_loader)
        dev_losses.append(dev_loss)
        if dev_loss <= min(dev_losses):
            torch.save(model.state_dict(), 'model.pt')
        print(f'Epoch: {epoch + 1:02} | Loss: {total_loss:.5f} | Dev Loss: {dev_loss:.5f}')


def test(args):

    print(args.corpus_path, args.vocab_path)

    txt_dataset = TextDataset(args.corpus_path, args.vocab_path)

    EMBEDDING_DIM = 600
    VOCAB_SIZE = txt_dataset.vocab_size
    HIDDEN_DIM = 600
    DROPOUT = 0.0
    NUM_LAYERS = 1

    BATCH_SIZE = 1
    NUM_WORKERS = 8

    train_set_size = int(0.7 * len(txt_dataset))
    dev_set_size = int(0.15 * len(txt_dataset))
    test_set_size = len(txt_dataset) - train_set_size - dev_set_size
    train_set, dev_set, test_set = random_split(txt_dataset, [train_set_size, dev_set_size, test_set_size])

    test_dataloader = DataLoader(train_set,
                                 batch_size=BATCH_SIZE,
                                 shuffle=True,
                                 num_workers=NUM_WORKERS,
                                 collate_fn=pad_sentences)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss(reduction='sum').to(device)
    print("Testing on device:", device)
    model = NeuralLanguageModel(NUM_LAYERS, VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, DROPOUT).to(device)
    if not os.path.exists(args.test):
        raise Exception(f'{args.test} not found')
    model.load_state_dict(torch.load(args.test))
    model.eval()
    with torch.no_grad():
        losses = []
        for batch in test_dataloader:
            input, target = batch
            input = input.to(device)
            target = target.to(device)
            output = model(input)
            target = torch.flatten(target)
            non_padding = (target != 3)
            output = output[non_padding]
            target = target[non_padding]
            loss = criterion(output, target)
            non_padding = (input[0] != 3)
            input = input[0][non_padding]
            input = ' '.join([txt_dataset.itos[i] for i in input.cpu().numpy()[1:]])
            loss = loss.cpu().numpy()
            losses.append(loss)
            print(f"{input}\t{loss}")
        avg = np.mean(losses)
        print("Average loss:", avg)


def test_one(args):
    print(args.corpus_path, args.vocab_path)

    txt_dataset = TextDataset(args.corpus_path, args.vocab_path)

    EMBEDDING_DIM = 400
    VOCAB_SIZE = txt_dataset.vocab_size
    HIDDEN_DIM = 1024
    DROPOUT = 0.5
    NUM_LAYERS = 2

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("testing on device:", device)
    model = NeuralLanguageModel(NUM_LAYERS, VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, DROPOUT).to(device)
    criterion = nn.CrossEntropyLoss()
    if not os.path.exists('model.pt'):
        raise Exception('model.pt not found')
    model.load_state_dict(torch.load('model.pt'))
    model.eval()
    with torch.no_grad():
        while True:
            sentence = input('Enter a sentence: ')
            if sentence == 'exit':
                break
            tokenizer = RegexTokenizer()
            sentence = tokenizer.tokenize(sentence)
            sentence = [txt_dataset.vocab.stoi[word] for word in sentence]
            sentence = torch.tensor(sentence).unsqueeze(0)
            sentence = sentence.to(device)
            output = model(sentence)
            output = torch.flatten(output, end_dim=1)
            preds = output.argmax(dim=1)
            print(preds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='NeuralLanguageModel.py',
        description='Predict the next word in a sentence.',
        epilog='author: Kannav Mehta')
    parser.add_argument('corpus_path')
    parser.add_argument('vocab_path')
    parser.add_argument('--train')
    parser.add_argument('--test')
    args = parser.parse_args()

    if args.train or args.test:
        if args.train:
            train(args)
        if args.test:
            test(args)
    else:
        test_one(args)
