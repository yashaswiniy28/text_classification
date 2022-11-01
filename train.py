import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import preprocess_data


def train_model(args, model, data_train, total_words, word2index):
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss=0
    # Train the Model
    for epoch in range(args.epochs):
        total_batch = int(len(data_train.data) / args.batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = preprocess_data.pre_processor.get_batch(data_train, i, args.batch_size, total_words, word2index)
            articles = Variable(torch.FloatTensor(batch_x))
            labels = Variable(torch.LongTensor(batch_y))

            optimizer.zero_grad()
            outputs = model(articles)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                      % (epoch+1, args.epochs, i + 1, len(data_train.data) // args.batch_size, loss.item()))

    save(model, args.save_dir, 'snapshot', len(data_train.data) // args.batch_size)


def save(model, save_dir, save_prefix, steps):
    print('Saving the model snapshot.......')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)
    print('Model snapshot is saved in mentioned folder: '+save_path)