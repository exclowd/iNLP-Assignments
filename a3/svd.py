import argparse
import os

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import ujson as json
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

SKIP_SENTENCES_AFTER = 40000

words = [
    'watch',
    'movie',
    'amazing',
    'she',
    'quickly',
    'from',
    'while'
]


def dump_vocab(data_path):
    file = os.path.join(data_path, 'reviews_Movies_and_TV.json')
    cnt = {}
    with open(file, 'r') as f:
        for i, line in tqdm(enumerate(f)):
            review = json.loads(line)['reviewText']
            review = nltk.word_tokenize(review)
            review = [word.lower() for word in review if word.isalpha()]
            for word in review:
                cnt[word] = cnt.get(word, 0) + 1
            if i > SKIP_SENTENCES_AFTER:
                break
    cnt = {k: v for k, v in cnt.items() if v >= 5}
    sorted_cnt = sorted(cnt.items(), key=lambda x: x[1], reverse=True)
    with open('vocab.txt', 'w+') as f:
        for word, cnt in sorted_cnt:
            f.write(f"{word}\t{cnt}\n")


def load_vocab(vocab):
    word2idx = {'<sos>': 0, '<eos>': 1}
    idx2word = {0: '<sos>', 1: '<eos>'}
    with open(vocab, 'r') as f:
        for idx, line in enumerate(f):
            word, _ = line.strip().split('\t')
            word2idx[word] = len(word2idx)
            idx2word[len(idx2word)] = word
    return word2idx, idx2word


def create_covariance_matrix(data_path, word2idx, seq_len=5):
    file = os.path.join(data_path, 'reviews_Movies_and_TV.json')
    vocab_size = len(word2idx)
    matrix = np.zeros((vocab_size, vocab_size), dtype=np.float32)
    with open(file, 'r') as f:
        for i, line in tqdm(enumerate(f)):
            review = json.loads(line)['reviewText']
            review = nltk.word_tokenize(review)
            review = ['<sos>'] + [word.lower() for word in review if word.isalpha()] + ['<eos>']
            for j in range(1, len(review) - 1):
                if review[j] not in word2idx:
                    continue
                prev = review[max(0, j - seq_len):j]
                after = review[j + 1:min(len(review), j + seq_len + 1)]
                co_occur = prev + after
                for word in co_occur:
                    if word not in word2idx:
                        continue
                    matrix[word2idx[review[j]], word2idx[word]] += 1
                    matrix[word2idx[word], word2idx[review[j]]] += 1
            if i > SKIP_SENTENCES_AFTER:
                break
    return matrix


def main(data_path, vocab=None):
    if vocab is None:
        dump_vocab(data_path)
        vocab = 'vocab.txt'
    word2idx, idx2word = load_vocab(vocab)
    print("Vocab size:", len(word2idx))
    matrix = create_covariance_matrix(data_path, word2idx)
    print(matrix.shape)
    reduced = TruncatedSVD(n_components=100, n_iter=300).fit_transform(matrix)
    # tsne plot
    mapping = {}
    neighbors = NearestNeighbors(n_neighbors=10, metric='cosine', algorithm='brute').fit(reduced)
    for word in words:
        idx = word2idx[word]
        distances, indices = neighbors.kneighbors(reduced[idx].reshape(1, -1))
        mapping[word] = indices[0]
    # visualize the mapping using TSNE and plot it using seaborn
    tsne = TSNE(n_components=2, perplexity=20)
    reduced_neighbors = {}
    for word in words:
        for idx in mapping[word]:
            reduced_neighbors[idx] = word
    reduced_neighbors = [(reduced[idx], word) for idx, word in reduced_neighbors.items()]
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
    plt.savefig(f'svd_tsne_{SKIP_SENTENCES_AFTER}.png')
    check_words = ['mother', 'father', 'titanic']
    for word in check_words:
        distances, indices = neighbors.kneighbors(reduced[word2idx[word]].reshape(1, -1))
        print(f"Neighbors of '{word}':")
        for i in indices[0]:
            print(f"{i}. {idx2word[i]}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, help='Path to the data directory')
    parser.add_argument('--vocab', type=str, help='use existing vocab file')
    args = parser.parse_args()
    main(args.data_path, args.vocab)
