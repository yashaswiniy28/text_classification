import preprocess_data
import torch
from torch.autograd import Variable

def test_model(data_test, model, args, total_words, word2index):
    # Test the Model
    correct = 0
    total = 0
    batch_x_test, batch_y_test = preprocess_data.pre_processor.get_batch(data_test, 0, args.batch_size, total_words, word2index)
    articles = Variable(torch.FloatTensor(batch_x_test))
    labels = torch.LongTensor(batch_y_test)
    outputs = model(articles)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

    print('Accuracy of the network on the 1180 test articles: %d %%' % (100 * correct / total))