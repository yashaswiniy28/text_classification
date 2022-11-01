## Text classification using PyTorch
Text classification model implemented with simple Multilayer Perceptron.

## Dataset
20 Newsgroups dataset is a collection of approximately 20,000 newsgroup documents, partitioned (nearly) evenly across 20 different newsgroups.
Available at [20 Newsgroups website](http://qwone.com/~jason/20Newsgroups/)

Note: newsgroup-related metadata such as 'headers', 'footers', 'quotes' has been removed while loading the dataset for realistic approach.

## Requirement
* python 3
* pytorch
* scikit-learn
* pandas
* numpy
## Usage
```
./main.py -h
```
or 
```
python main.py -h
```
or
```
python3 main.py -h
```

You can see the below optional arguments:
```
Text classifier for 20 Newsgroups dataset

options:
  -h, --help            show this help message and exit
  -lr LR                initial learning rate [default: 0.001]
  -epochs EPOCHS        number of epochs for train [default: 10]
  -batch-size BATCH_SIZE
                        batch size for training [default: 64]
  -all_categories       all or few categories
  -snapshot SNAPSHOT    filename of model snapshot [default: None]
  -test                 train or test
  -save-dir SAVE_DIR    where to save the snapshot
```

## Train
By default, model is trained on executing main file.
```
./main.py
```
or 
```
python main.py
```
or
```
python3 main.py
```

## Save model
This will save the model snapshots to the mentioned directory
```
 python main.py -save-dir snapshots/2022-11-01
```

## Test
```
 python main.py -test -snapshot="./snapshot/snapshot_steps_31.pt"
```
or 
```
 python main.py -all-categories -test -snapshot="./snapshot/snapshot_steps_176.pt"
```
Important: You can test the model from the previously saved, by passing the -snapshot argument.

## Performance
1. Model trained on 4 categories obtained test accuracy ~ 71%
2. Model trained on all categories obtained test accuracy ~ 82%
