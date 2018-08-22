from collections import Counter

import pandas as pd


def read_data(basename, names=None):
    return pd.read_csv('../data/clickstream/{}.csv'.format(basename),
                       sep=' ', names=names)


class Tree:
    def __init__(self):
        self.parent = None
        self.children = {}
        self.attribute = None
        self.attr_values = None
        self.label = None

    def _best_attr(self, features, labels):
        pass

    def induce(self, features, labels, alpha):
        classes = Counter(labels)
        if len(classes) == 1 or features.shape[1] == 0:
            self.label = classes.most_common(1)[0]
        else:
            self.attribute = self._best_attr(features, labels)
            self.attr_values = features[self.attribute].unique()
            for val in self.attr_values:
                sub_features = features


if __name__ == '__main__':
    # load the data
    with open('../data/clickstream/featnames.csv') as inp:
        names = list(l.strip() for l in inp)
    trainfeat = read_data('trainfeat', names)
    trainlabs = read_data('trainlabs', ['result'])
    testfeat = read_data('testfeat', names)
    testlabs = read_data('testlabs', ['result'])
    print(trainfeat["Session First Request Hour of Day Bin"])
