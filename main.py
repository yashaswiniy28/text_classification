import torch
from torch.autograd import Variable
import torch.nn as nn
import argparse
import dataset_loader as dl
import model_mlp as MLP
import test
import train
from collections import Counter
import preprocess_data


parser = argparse.ArgumentParser(description='Text classifier for 20 Newsgroups dataset')
# model learning inputs
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=10, help='number of epochs for train [default: 10]')
parser.add_argument('-batch-size', type=int, default=64, help='batch size for training [default: 64]')
parser.add_argument('-all-categories', action='store_true', default=False, help='all or few categories')

# options
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-test', action='store_true', default=False, help='train or test')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
args = parser.parse_args()


############################
data_train, data_test = dl.load_dataset.load_20newsgrop(args)
print('Dataset loaded successfully')
target_names = data_train.target_names
y_train, y_test = data_train.target, data_test.target

# mapping word to integer
vocab = Counter()
for text in data_train.data:
    for word in text.split(' '):
        vocab[word.lower()] += 1

for text in data_test.data:
    for word in text.split(' '):
        vocab[word.lower()] += 1

total_words = len(vocab)
word2index = preprocess_data.pre_processor.get_word_2_index(vocab)

# initialize tensors and model parameters
loss = nn.CrossEntropyLoss()
input = Variable(torch.randn(2, 5), requires_grad=True)
print('Input tensor: ',input)
target = Variable(torch.LongTensor(2).random_(5))
print('Target tensor: ', target)
output = loss(input, target)
output.backward()
hidden_size = 100
input_size = total_words
if args.all_categories:
    num_classes = 20
else:
    num_classes = 4

#initialize the model
model = MLP.NeurNet(input_size, hidden_size, num_classes)

# train or test the model
if args.test:
    if args.snapshot is not None:
        try:
            print()
            print('\nLoading model from {}...'.format(args.snapshot))
            model.load_state_dict(torch.load(args.snapshot))
            print('Model evaluation started')
            test.test_model(data_test, model, args, total_words, word2index)
            print('Model evaluation completed')
        except Exception as e:
            print(e)
    else:
        print()
        print('Missing argument -snapshot. Could not test the model without mentioning previously trained model! Try to execute without -test argument to train and test the model. \n')

else:
    print()
    try:
        print('Loading the model.....')
        print('Model training started.......')
        train.train_model(args, model, data_train, total_words, word2index)
        print('Model training completed')
        print('Model evaluation started')
        test.test_model(data_test, model, args, total_words, word2index)
        print('Model evaluation completed')
    except Exception as e:
        print(e)
