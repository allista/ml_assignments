import json
from collections import Counter

import pandas as pd
from scipy.stats import chisquare, entropy


class Tree:
    def __init__(self, parent=None, attribute=None, label=None):
        self.parent = parent
        self.children = {}
        self.attribute = attribute
        self.label = label

    @staticmethod
    def _split(features, labels, attribute):
        subs = {}
        attr = features[attribute]
        for val in attr.unique():
            sub_features = features[attr == val].drop(columns=[attribute])
            sub_labels = labels.loc[sub_features.index]
            subs[val] = (sub_features, sub_labels, Counter(sub_labels))
        return subs

    @classmethod
    def _best_attr(cls, features, labels, classes, alpha):
        best_attr = [-1, None, None]
        S = entropy(list(classes.values()))
        total = float(len(labels))
        for attr in features:
            gain = S
            subs = cls._split(features, labels, attr)
            # if cls._should_stop(labels, classes, subs, alpha):
            #     continue # 0.26584
            for val, (sub_f, sub_l, sub_c) in subs.items():
                gain -= len(sub_l) / total * entropy(list(sub_c.values()))
            if gain > best_attr[0]:
                best_attr = [gain, attr, subs]
        if cls._should_stop(labels, classes, best_attr[2], alpha):
            return -1, None, None  # 0.25376
        return best_attr

    @staticmethod
    def _should_stop(labels, classes, subs, alpha):
        total = float(len(labels))
        cnames = sorted(classes)
        observed = []
        expected = []
        for val, (sub_features, sub_labels, sub_count) in subs.items():
            sub_len = len(sub_labels)
            observed.append([sub_count[k] for k in cnames])
            expected.append([classes[k] * sub_len / total for k in cnames])
        chi, pv = chisquare(observed, expected, axis=None)
        return pv >= alpha

    def _subtree(self, val):
        sub_tree = Tree(self)
        self.children[val] = sub_tree
        return sub_tree

    def induce(self, features, labels, classes=None, alpha=0.05):
        if classes is None:
            classes = Counter(labels)
        self.label = classes.most_common(1)[0][0]
        if len(classes) == 1 or features.shape[1] == 0:
            print('Stopped with the most common class: {}'
                  .format(classes.most_common()))
        else:
            gain, self.attribute, subs = self._best_attr(features, labels, classes, alpha)
            if self.attribute is None:
                print('Stopped by Chi2 with the most common class: {}'
                      .format(classes.most_common()))
            else:
                print('Splitting at "{}" with gain {}'.format(self.attribute, gain))
                for val, (sub_features, sub_labels, sub_classes) in subs.items():
                    self._subtree(val).induce(sub_features, sub_labels, sub_classes, alpha)

    def evaluate(self, row):
        if not self.children:
            return self.label
        val = row[self.attribute]
        child = self.children.get(val)
        if child:
            return child.evaluate(row)
        return self.label

    def error_rate(self, features, labels):
        total = len(labels)
        return sum(self.evaluate(features.iloc[i]) != labels.iloc[i]
                   for i in range(total)) / float(total)

    def for_json(self):
        obj = {'label': self.label}
        if self.children:
            obj['attribute'] = self.attribute
            obj['children'] = {str(k): v.for_json() for k, v in self.children.items()}
        return obj

    def __str__(self):
        if not self.children:
            return str(self.label)
        return '("{}": [{}])'.format(self.attribute,
                                     ', '.join('{}: {!s}'.format(v, t)
                                               for v, t in self.children.items()))

    def __repr__(self):
        return str(self)

    def save(self, filename):
        with open(filename, 'w') as out:
            json.dump(self.for_json(), out, indent=2, sort_keys=True)

    @classmethod
    def from_object(cls, obj, parent=None):
        node = cls(parent,
                   attribute=obj.get('attribute', None),
                   label=obj.get('label', None))
        for val, child in obj.get('children', {}).items():
            node.children[int(val)] = cls.from_object(child, node)
        return node

    @classmethod
    def load(cls, filename):
        with open(filename) as inp:
            return cls.from_object(json.load(inp))


if __name__ == '__main__':
    import os


    def read_data(basename, names=None):
        return pd.read_csv('../data/clickstream/{}.csv'.format(basename),
                           sep=' ', names=names)


    names = read_data('featnames', ['name'])
    trainfeat = read_data('trainfeat', names.name)
    trainlabs = read_data('trainlabs', ['result'])
    testfeat = read_data('testfeat', names.name)
    testlabs = read_data('testlabs', ['result'])

    savefile = 'clickstream-005.json'
    if os.path.isfile(savefile):
        t = Tree.load(savefile)
    else:
        t = Tree()
        t.induce(trainfeat, trainlabs.result, alpha=0.05)
        t.save(savefile)
    print('Train error rate: {}'.format(t.error_rate(trainfeat, trainlabs.result)))
    print('Test error rate: {}'.format(t.error_rate(testfeat, testlabs.result)))
