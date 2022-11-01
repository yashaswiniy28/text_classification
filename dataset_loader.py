from sklearn.datasets import fetch_20newsgroups
"""
this file contains class to load the dataset.
newsgroup-related metadata such as 'headers', 'footers', 'quotes' has been removed while loading the dataset
- option to load all categories or only selected 4 categories.

"""
class load_dataset():
    def load_20newsgrop(args):
        
        if args.all_categories:
            categories = None
        else:
            categories = [
                'sci.electronics',
                'alt.atheism',
                'talk.religion.misc',
                'comp.graphics'
            ]

        remove = ('headers', 'footers', 'quotes')

        print("Loading 20 newsgroups dataset for categories:")
        print(categories if categories else "all")

        data_train = fetch_20newsgroups(subset='train', remove=remove, categories=categories,
                                shuffle=True, random_state=42)

        data_test = fetch_20newsgroups(subset='test', remove=remove, categories=categories,
                               shuffle=True, random_state=42)
        print('total texts in train:', len(data_train.data))
        print('total texts in test:', len(data_test.data))
        return data_train, data_test
