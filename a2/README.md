# Neural POS Tagger

## How to Run
- First extract the dataset into the `${ROOT_DIR}/data` directory
- To train the model on the dataset
```bash
python pos_tagger.py --train=true
```
- To test the model on the dataset and generate the classification report
```bash
python pos_tagger.py --test=true
```

- To test the model on a single sentence and generate the tags
```bash
python pos_tagger.py one