import numpy as np
""" 
this class consists of necessary data preperation functions.
data is pre-processed before feeding to the neural network.
"""

class pre_processor():

    # map the word with integer
    def get_word_2_index(vocab):
        word2index = {}
        for i, word in enumerate(vocab):
            word2index[word.lower()] = i

        return word2index


    # prepare the data batch of size 'batch_size'
    def get_batch(df, i, batch_size, total_words, word2index):
        batches = []
        results = []
        texts = df.data[i * batch_size:i * batch_size + batch_size]
        categories = df.target[i * batch_size:i * batch_size + batch_size]
        for text in texts:
            layer = np.zeros(total_words, dtype=float)
            for word in text.split(' '):
                layer[word2index[word.lower()]] += 1

            batches.append(layer)

        for category in categories:
            if category == 0:
                index_y = 0
            elif category == 1:
                index_y = 1
            else:
                index_y = 2
            results.append(index_y)

        return np.array(batches), np.array(results)