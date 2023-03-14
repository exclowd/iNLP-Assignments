import argparse
import math
from typing import List, Tuple

import numpy as np
import regex
from sklearn.model_selection import train_test_split


class RegexTokenizer:
    def __init__(self):
        self.quote_rx = regex.compile(r"[—_\"\*\[\]]")
        self.corpus_rx = [(regex.compile(r"Mrs."), "Mrs"),
                          (regex.compile(r"Mr."), "Mr")]
        self.line_break_rx = regex.compile(r"(?<!\w\.\w.)(?<![A-Z]\.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!|\:)\s+")
        self.hashtag_rx = regex.compile(r"\B(\#[a-zA-Z]+\b)(?!;)")
        self.mention_rx = regex.compile(r"\B(\@[a-z_A-Z]+\b)(?!;)")
        self.url_rx = regex.compile(
            r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)" +
            r"(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|" +
            r"(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))")
        self.whitespace_rx = regex.compile(r"[\s<\w>\d]")
        self.tokenize_rx = regex.compile(
            r"[a-zA-Z<>’]+(?:(?=\s)|(?=\:\s)|(?=$)|(?=[.!,:?;\”]))")

    def tokenize(self, string: str, pad_sentences: bool = True) -> List[List[str]]:
        s = self.quote_rx.sub(' ', string)
        for r, c in self.corpus_rx:
            s = r.sub(c, s)
        c = self.line_break_rx.sub('\1', s)
        sentences = c.split('\1')
        out = []
        for s in sentences:
            s = self.url_rx.sub('<URL>', s)
            s = self.hashtag_rx.sub('<HASHTAG>', s)
            tokens = self.tokenize_rx.findall(s.lower())
            if pad_sentences:
                tokens = ['<start>'] + tokens + ['<end>']
            out.append(tokens)
        return out

    def split_sentences(self, string: str) -> List[str]:
        s = self.quote_rx.sub(' ', string)
        for r, c in self.corpus_rx:
            s = r.sub(c, s)
        c = self.line_break_rx.sub('\1', s)
        sentences = c.split('\1')
        return sentences


def get_ngrams(sentence: List[str], n):
    for i in range(len(sentence) - n + 1):
        yield tuple(sentence[i: i + n])


class KNSmoothingLM:
    def __init__(self, n=4, d=0.75):
        self.cnt = None
        self.vocab = None
        self.vocab_size = None
        self.uniq_suff = None
        self.cont_cnt_sum = None
        self.cnt_sum = None
        self.cont_cnt = None
        self.N = n
        self.d = d
        self.UNK = 10
        self.removed_words = []

    def fit(self, sentences) -> None:
        # Things needed:
        self.vocab = ['<unk>']
        for s in sentences:
            for w in s:
                if w not in self.vocab:
                    self.vocab.append(w)
        self.vocab_size = len(self.vocab)
        # 1. count for every ngram
        self.cnt = [{} for _ in range(self.N + 1)]
        # represent each w as a tuple (w,)
        for s in sentences:
            for w in s:
                self.cnt[1][(w,)] = self.cnt[1].get((w,), 0) + 1
        # Now here is a heuristic I don't even understand
        # kneser and ney are a bunch of fucks
        sorted_cnts = dict(sorted(self.cnt[1].items(), key=lambda x: x[1], reverse=True))
        for _ in range(self.UNK):
            w, _ = sorted_cnts.popitem()
            self.removed_words.append(w[0])
        self.removed_words = set(self.removed_words)
        for w in self.removed_words:
            self.vocab.remove(w)
        sentences = [['<unk>' if w in self.removed_words else w for w in s] for s in sentences]
        self.cnt[1] = sorted_cnts

        self.cnt[1][('<unk>',)] = self.UNK

        for s in sentences:
            for n in range(2, self.N + 1):
                for ngram in get_ngrams(s, n):
                    self.cnt[n][ngram] = self.cnt[n].get(ngram, 0) + 1
        # 3. continuation count for every ngram, unique single word prefixes for every ngram
        self.cont_cnt = [{} for _ in range(self.N + 1)]
        for n in range(2, self.N + 1):
            for ngram in self.cnt[n]:
                context, string = ngram[:1], ngram[1:]
                self.cont_cnt[n - 1][string] = self.cont_cnt[n - 1].get(string, set())
                self.cont_cnt[n - 1][string].add(context)
        for n in range(1, self.N):
            for ngram in self.cont_cnt[n]:
                self.cont_cnt[n][ngram] = len(self.cont_cnt[n][ngram])
        # 4. sum of continuation counts of all unique single word prefixes for every ngram
        self.cont_cnt_sum = [{} for _ in range(self.N + 1)]
        for n in range(2, self.N):
            for ngram in self.cont_cnt[n]:
                prefix = ngram[:-1]
                self.cont_cnt_sum[n - 1][prefix] = self.cont_cnt_sum[n - 1].get(prefix, 0) + self.cont_cnt[n][ngram]
        # 5. unique single word suffixes for every ngram
        self.uniq_suff = [{} for _ in range(self.N + 1)]
        for n in range(2, self.N + 1):
            for ngram in self.cnt[n]:
                string, suffix = ngram[:-1], ngram[-1]
                self.uniq_suff[n - 1][string] = self.uniq_suff[n - 1].get(string, set())
                self.uniq_suff[n - 1][string].add(suffix)
        for n in range(1, self.N):
            for ngram, cnt in self.uniq_suff[n].items():
                self.uniq_suff[n][ngram] = len(cnt)

    def P(self, ngram: Tuple[str]) -> float:
        EPS = 0.00001
        n = len(ngram)
        context = ngram[:-1]
        if n == 1:
            num_bigrams = len(self.cnt[2])
            return self.cont_cnt[1][ngram] / num_bigrams
        if ngram in self.cnt[n]:
            if len(ngram) == 4:
                A = max(self.cnt[n].get(ngram, 0) - self.d, 0) / self.cnt[n - 1].get(context, EPS)
                L = self.d * self.uniq_suff[n - 1].get(context, 0) / self.cnt[n - 1].get(context, EPS)
                B = L * self.P(ngram[1:])
                return A + B
            else:
                # here n < 4
                A = max(self.cont_cnt[n].get(ngram, 0) - self.d, 0) / self.cont_cnt_sum[n - 1].get(context, EPS)
                L = self.d * self.uniq_suff[n - 1].get(context, 0) / self.cnt[n - 1].get(context, EPS)
                B = L * self.P(ngram[1:])
                return A + B
        else:
            return self.P(ngram[1:])

    def calculate_perplexity(self, sentences: List[List[str]]):
        ans = np.zeros(len(sentences))
        for i, sent in enumerate(sentences):
            total = 0
            sent = sent[1:-1]  # remove end and start tokens
            sent = [w if w in self.vocab else '<unk>' for w in sent]
            if len(sent) == 0:
                ans[i] = 0
                continue
            n = min(len(sent), self.N)
            for ngram in get_ngrams(sent, n):
                p = self.P(ngram)
                total += math.log2(p)
            ans[i] = pow(2, (-total / len(sent)))
        return ans


def kneser_ney_lm(train_data, test_data, train=False, test=False):
    train_tokens = [x[1] for x in train_data]
    train_sent = [x[0] for x in train_data]
    test_tokens = [x[1] for x in test_data]
    test_sent = [x[0] for x in test_data]
    model = KNSmoothingLM()
    model.fit(train_tokens)
    if test:
        pp = model.calculate_perplexity(test_tokens)
        idx = np.argsort(pp)
        idx = idx[:int(0.95 * len(pp))]
        print("Avg:", np.mean(pp[idx]))
        for i in idx:
            print(f"{repr(test_sent[i])}\t{pp[i]}")
    elif train:
        pp = model.calculate_perplexity(train_tokens)
        idx = np.argsort(pp)
        idx = idx[:int(0.95 * len(idx))]
        print("Avg:", np.mean(pp[idx]))
        for i in idx:
            print(f"{repr(train_sent[i])}\t{pp[i]}")
    else:
        inp = input("Enter sentence: ")
        while inp != "exit":
            tokenizer = RegexTokenizer()
            tokens = tokenizer.tokenize(inp, pad_sentences=True)
            print(model.calculate_perplexity(tokens))
            inp = input("Enter sentence: ")


class WBSmoothingLM:
    def __init__(self, vocab_size, n=4):
        self.N = n
        self.cnt = [{} for _ in range(self.N + 1)]
        self.cont_cnt = [{} for _ in range(self.N + 1)]
        self.vocab = None
        self.vocab_size = vocab_size

    def fit(self, sentences: List[List[str]]) -> None:
        self.vocab = ['<unk>']
        for sent in sentences:
            for w in sent:
                if w not in self.vocab:
                    self.vocab.append(w)
        # 1. count for every ngram
        for sent in sentences:
            for w in sent:
                self.cnt[1][(w,)] = self.cnt[1].get((w,), 0) + 1

        for s in sentences:
            for n in range(2, self.N + 1):
                for ngram in get_ngrams(s, n):
                    self.cnt[n][ngram] = self.cnt[n].get(ngram, 0) + 1
        # 3. continuation count for every ngram, unique single word prefixes for every ngram
        self.cont_cnt = [{} for _ in range(self.N + 1)]
        for n in range(2, self.N + 1):
            for ngram in self.cnt[n]:
                context, string = ngram[:-1], ngram[-1]
                self.cont_cnt[n - 1][context] = self.cont_cnt[n - 1].get(context, set())
                self.cont_cnt[n - 1][context].add(string)
        for n in range(1, self.N):
            for ngram in self.cont_cnt[n]:
                self.cont_cnt[n][ngram] = len(self.cont_cnt[n][ngram])
        self.ZGHC = sum(self.cnt[1].values())
        self.N1pE = len(self.vocab)

    def P(self, ngram: Tuple[str]) -> float:
        n = len(ngram)
        if n == 1:
            A = self.ZGHC / (self.ZGHC + self.N1pE) * (self.cnt[1].get(ngram, 0) / self.ZGHC)
            B = self.N1pE / (self.ZGHC + self.N1pE) * (1 / self.vocab_size)
            return A + B
        else:
            prefix = ngram[:-1]
            if prefix in self.cont_cnt[n-1]:
                A = self.cnt[n-1].get(prefix, 0)
                A /= (self.cnt[n-1].get(prefix, 0.00001) + self.cont_cnt[n-1].get(prefix, 0))
                A *= self.cnt[n].get(ngram, 0) / self.cnt[n-1].get(prefix, 1)
                B = self.cont_cnt[n-1].get(prefix, 0)
                B /= (self.cnt[n-1].get(prefix, 0.00001) + self.cont_cnt[n-1].get(prefix, 0))
                B *= self.P(ngram[1:])
                return A + B
            else:
                return self.P(ngram[1:])

    def calculate_perplexity(self, sentences: List[List[str]]) -> np.ndarray:
        ans = np.zeros(len(sentences))
        for i, sent in enumerate(sentences):
            total = 0
            sent = sent[1:-1]
            if len(sent) == 0:
                ans[i] = 0
                continue
            n = min(len(sent), self.N)
            for ngram in get_ngrams(sent, n):
                p = self.P(ngram)
                total += math.log2(p)
            ans[i] = pow(2, (-total / len(sent)))
        return ans


def witten_bell_lm(train_data, test_data, train=False, test=False):
    train_tokens = [x[1] for x in train_data]
    train_sent = [x[0] for x in train_data]
    test_tokens = [x[1] for x in test_data]
    test_sent = [x[0] for x in test_data]
    vocab_size = set([w for sent in train_tokens for w in sent])
    for sent in test_tokens:
        for w in sent:
            if w not in vocab_size:
                vocab_size.add(w)
    model = WBSmoothingLM(vocab_size=len(vocab_size))
    model.fit(train_tokens)
    if test:
        pp = model.calculate_perplexity(test_tokens)
        idx = np.argsort(pp)
        idx = idx[:int(0.95 * len(pp))]
        print("Avg:", np.mean(pp[idx]))
        for i in idx:
            print(f"{repr(test_sent[i])}\t{pp[i]}")
    elif train:
        pp = model.calculate_perplexity(train_tokens)
        idx = np.argsort(pp)
        idx = idx[:int(0.95 * len(idx))]
        print("Avg:", np.mean(pp[idx]))
        for i in idx:
            print(f"{repr(train_sent[i])}\t{pp[i]}")
    else:
        inp = input("Enter sentence: ")
        while inp != "exit":
            tokenizer = RegexTokenizer()
            tokens = tokenizer.tokenize(inp, pad_sentences=True)
            print(model.calculate_perplexity(tokens))
            inp = input("Enter sentence: ")


def main(args):
    with open(args.corpus_path, 'r') as f:
        tokenizer = RegexTokenizer()
        raw = f.read()
        tokens = tokenizer.tokenize(raw, pad_sentences=True)
        sentences = tokenizer.split_sentences(raw)

    data = [(s, t) for s, t in zip(sentences, tokens)]
    # TODO change this later
    train, test = train_test_split(data, test_size=1000, random_state=23)

    if args.smoothing_type == "k":
        kneser_ney_lm(train, test, train=args.train, test=args.test)
    elif args.smoothing_type == "w":
        witten_bell_lm(train, test, train=args.train, test=args.test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='language_model.py',
        description='LM using KN smoothing and WB smoothing',
        epilog='author: Kannav Mehta'
    )
    parser.add_argument('smoothing_type')
    parser.add_argument('corpus_path')
    parser.add_argument('--train')
    parser.add_argument('--test')
    arguments = parser.parse_args()
    main(arguments)
