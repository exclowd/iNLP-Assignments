The python files obey the semantics mentioned in the assignment pdf
models are in https://drive.google.com/drive/folders/1_4Gj91T3oclnRNF1uiyFaiN9UrghLpfR?usp=share_link

Experiments
1. Experimented with the d for KN smoothing
    d = 0.5 performs worse than d = 0.75
2. KN performs better than WB smoothing
3. Experimented with differnet types of tokenization
4. Experimented with LSTM layers and parameters(tuning)

# neural_language_model

## How to Run
- First extract the dataset into the `${ROOT_DIR}/data` directory
- To train the model on the dataset
```
  python neural_language_model.py ./data/ulysses.txt ./vocab/vocab_ulysses.txt --train=true
  python neural_language_model.py ./data/pride.txt ./vocab/vocab_pride.txt --train=true
```

- To test the model on dataset
```
  python neural_language_model.py ./data/pride.txt ./vocab/vocab_pride.txt --test=./pride.pt > 2019101044_LM5_train-perplexity.txt
  python neural_language_model.py ./data/ulysses.txt ./vocab/vocab_ulysses.txt --test=./ulysses.pt > 2019101044_LM6_test-perplexity.txt
```
`
## How to Run
- First extract the dataset into the `${ROOT_DIR}/data` directory
- To run the model train dataset
```
python language_model.py k ./data/ulysses.txt --train=true > 2019101044_LM3_train-perplexity.txt
python language_model.py w ./data/ulysses.txt --train=true > 2019101044_LM4_train-perplexity.txt
```

- To run the model test dataset
```
python language_model.py k ./data/ulysses.txt --test=true > 2019101044_LM3_train-perplexity.txt
python language_model.py w ./data/ulysses.txt --test=true > 2019101044_LM4_train-perplexity.txt
```
