import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from nltk import word_tokenize
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm

SKIP_SENTENCES_AFTER = 40000


class MoviesAndTVDataset(Dataset):
    def __init__(self, data_path, vocab=None, seq_len=6):
        self.data_path = data_path
        self.file = os.path.join(data_path, 'reviews_Movies_and_TV.json')
        if vocab is None:
            self._dump_vocab(thresh=5)
            self.vocab = 'vocab.txt'
        self.word2idx = {}
        self.idx2word = {}
        self.wordcnt = {}
        self.load_vocab(self.vocab)
        self.vocab_size = len(self.word2idx)
        self.removed_words = []
        self.do_negative_sampling()
        self.SEQ_LEN = seq_len
        self.X = []
        self.y = []
        self.load_data()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def _dump_vocab(self, thresh=2):
        cnt = {}
        with open(self.file, 'r') as f:
            for i, line in tqdm(enumerate(f)):
                review = json.loads(line)['reviewText']
                review = word_tokenize(review)
                review = [word.lower() for word in review if word.isalpha()]
                for word in review:
                    cnt[word] = cnt.get(word, 0) + 1
                if i > SKIP_SENTENCES_AFTER:
                    break
        cnt = {k: v for k, v in cnt.items() if v >= thresh}
        sorted_cnt = sorted(cnt.items(), key=lambda x: x[1], reverse=True)
        with open('vocab.txt', 'w+') as f:
            for word, cnt in sorted_cnt:
                f.write(f"{word}\t{cnt}\n")

    def load_vocab(self, vocab):
        word2idx = {'<unk>': 0, '<sos>': 1, '<eos>': 2}
        idx2word = {0: '<unk>', 1: '<sos>', 2: '<eos>'}
        wordcnt = {}
        with open(vocab, 'r') as f:
            for idx, line in enumerate(f):
                word, cnt = line.strip().split('\t')
                wordcnt[word] = int(cnt)
                word2idx[word] = len(word2idx)
                idx2word[len(idx2word)] = word
        self.word2idx, self.idx2word = word2idx, idx2word
        self.wordcnt = wordcnt

    def do_negative_sampling(self):
        total = sum([pow(x, 0.75) for x in self.wordcnt.values()])
        negatives = []
        probs = []
        print(total)
        for word, cnt in self.wordcnt.items():
            negatives.append(self.word2idx[word])
            probs.append(pow(cnt, 0.75) / total)
        negatives = np.array(negatives)
        probs = np.array(probs)
        removed = np.random.choice(negatives, size=20, p=probs)
        self.removed_words = [word for word in removed]
        print("Removed words: ")
        print([self.idx2word[word] for word in removed])

    def load_data(self):
        with open(self.file, 'r') as f:
            for i, line in tqdm(enumerate(f)):
                review = json.loads(line)['reviewText']
                review = word_tokenize(review)
                review = [word.lower() for word in review if word.isalpha()]
                review = [self.word2idx.get(word, self.word2idx['<unk>']) for word in review]
                # print(review)
                if len(review) < self.SEQ_LEN:
                    continue
                for j in range(len(review) - 2 * self.SEQ_LEN):
                    seq = review[j:j + self.SEQ_LEN * 2 + 1]
                    inp = seq[:self.SEQ_LEN] + seq[self.SEQ_LEN + 1:]
                    # inp =  seq
                    out = seq[self.SEQ_LEN]
                    if seq[self.SEQ_LEN] < 20:
                        continue
                    if seq[self.SEQ_LEN] in self.removed_words:
                        out = 0
                    self.X.append(inp)
                    self.y.append(out)
                if i > SKIP_SENTENCES_AFTER:
                    break


class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_norm=None):
        super(CBOW, self).__init__()
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            max_norm=max_norm)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = torch.mean(x, dim=1)
        x = self.linear(x)
        return x


def batch_collate(batch):
    X, y = zip(*batch)
    X = torch.tensor(X)
    y = torch.tensor(y)
    return X, y


LEARNING_RATE = 0.01
EPOCHS = 25


def batch_sgd(model, train_loader, val_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses = []
    val_losses = []
    for epoch in tqdm(range(EPOCHS)):

        model.train()
        epoch_train_loss = 0
        for i, batch in enumerate(train_loader):
            X, y = batch
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_pred = model(X)
            preds = torch.argmax(y_pred, dim=1)
            # print(preds)
            # print(y)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            # input()

        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                X, y = batch
                X = X.to(device)
                y = y.to(device)
                y_pred = model(X)
                loss = criterion(y_pred, y)
                loss_val = loss.item()
                epoch_val_loss += loss_val

        epoch_train_loss /= len(train_loader)
        epoch_val_loss /= len(val_loader)

        print(f"Epoch {epoch + 1} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        # save model if val loss is minimum
        if epoch_val_loss == min(val_losses):
            torch.save(model.state_dict(), 'model_40000.pt')
            print(f"Model saved epoch:{epoch + 1}")

    return model, train_losses, val_losses


def embeddings_visualize(model_path, data_path, vocab=None):
    words = [
        'watch',
        'movie',
        'amazing',
        'she',
        'quickly',
        'from',
        'while'
    ]

    dataset = MoviesAndTVDataset(data_path, vocab)
    if not os.path.exists(model_path):
        raise Exception("Model not found")
    VOCAB_SIZE = dataset.vocab_size
    model = CBOW(VOCAB_SIZE, EMBEDDING_DIM, max_norm=MAX_NORM)
    model.load_state_dict(torch.load(model_path))
    embeddings = model.embedding.weight.detach().numpy()
    # norms = (embeddings ** 2).sum(axis=1) ** (1 / 2)
    # norms = np.reshape(norms, (len(norms), 1))
    # reduced = embeddings / norms

    reduced = embeddings

    mapping = {}
    neighbors = NearestNeighbors(n_neighbors=20,
                                 metric='cosine',
                                 algorithm='brute').fit(reduced)
    for word in words:
        idx = dataset.word2idx[word]
        distances, indices = neighbors.kneighbors(reduced[idx].reshape(1, -1))
        mapping[word] = indices[0]

    check_words = ['mother', 'father', 'titanic']

    for word in check_words:
        distances, indices = neighbors.kneighbors(reduced[dataset.word2idx[word]].reshape(1, -1))
        print(f"Neighbors of '{word}':")
        for i in indices[0]:
            print(f"{i}. {dataset.idx2word[i]}")
    # visualize the mapping using TSNE and plot it using seaborn
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    reduced_neighbors = [(reduced[idx], word) for word in words for idx in mapping[word]]
    tsne_results = tsne.fit_transform(np.array([x[0] for x in reduced_neighbors]))
    df = pd.DataFrame(columns=['x', 'y', 'color'])
    for i, (base_word, tsne_result) in enumerate(zip(reduced_neighbors, tsne_results)):
        df.loc[i] = [tsne_result[0], tsne_result[1], base_word[1]]
    # show the word from mapping in legend
    plt.figure(figsize=(8, 8))
    sns.scatterplot(
        x="x", y="y",
        hue="color",
        palette=sns.color_palette("bright", len(words)),
        data=df,
        legend="full",
        alpha=0.7
    )
    plt.savefig(f'tsne-cbow-{SKIP_SENTENCES_AFTER}.png')


BATCH_SIZE = 256
EMBEDDING_DIM = 200
MAX_NORM = 1


def test(model_path, data_path, vocab=None):
    dataset = MoviesAndTVDataset(data_path, vocab)
    train_len = int(0.7 * len(dataset))
    val_len = int(0.15 * len(dataset))
    test_len = len(dataset) - train_len - val_len
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_len, val_len, test_len])
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True,
                                              num_workers=4, collate_fn=batch_collate)
    if not os.path.exists(model_path):
        raise Exception("Model not found")
    VOCAB_SIZE = dataset.vocab_size
    model = CBOW(VOCAB_SIZE, EMBEDDING_DIM, max_norm=MAX_NORM)
    model.load_state_dict(torch.load(model_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    pred = []
    y_true = []
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            X, y = batch
            X = X.to(device)
            y = y.to(device)
            y_pred = model(X)
            y_pred = torch.argmax(y_pred, dim=1)
            pred.append(y_pred.cpu().numpy())
            y_true.append(y.cpu().numpy())

    pred = np.concatenate(pred, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    print(pred, y_true)
    print(pred.shape, y_true.shape)

    score = accuracy_score(y_true, pred)
    print(f"Test Accuracy: {score}")


def train(data_path, vocab=None):
    dataset = MoviesAndTVDataset(data_path, vocab)
    train_len = int(0.7 * len(dataset))
    val_len = int(0.2 * len(dataset))
    test_len = len(dataset) - train_len - val_len
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_len, val_len, test_len])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                                               num_workers=4, collate_fn=batch_collate)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True,
                                             num_workers=4, collate_fn=batch_collate)
    VOCAB_SIZE = dataset.vocab_size
    print(f"Vocab Size: {VOCAB_SIZE}")
    print(f"Train Size: {len(train_set)}")
    print(f"Val Size: {len(val_set)}")
    print(f"Test Size: {len(test_set)}")
    model = CBOW(VOCAB_SIZE, EMBEDDING_DIM, MAX_NORM)
    model, train_losses, val_losses = batch_sgd(model, train_loader, val_loader)
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Train a word2vec model using the CBOW architecture')
    parser.add_argument('data_path', type=str, default='./data')
    parser.add_argument('--vocab', type=str, default=None)
    parser.add_argument('--testmodel', type=str, default=None)
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()
    if args.testmodel:
        test(args.testmodel, args.data_path, args.vocab)
        if args.visualize:
            embeddings_visualize(args.testmodel, args.data_path, args.vocab)
    else:
        train(args.data_path, args.vocab)
