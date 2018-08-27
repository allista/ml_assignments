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
    def _split(features, label, attribute):
        subs = {}
        attr = features[attribute]
        for val in attr.unique():
            sub_features = features[attr == val]
            subs[val] = (sub_features, Counter(sub_features[label]))
        return subs

    @classmethod
    def _best_attr(cls, used_attrs, features, label, classes, alpha):
        best_attr = (-1, None, None)
        S = entropy(list(classes.values()))
        total = float(len(features.index))
        for attr in features:
            if attr in used_attrs:
                continue
            gain = S
            subs = cls._split(features, label, attr)
            for val, (sub_f, sub_c) in subs.items():
                gain -= len(sub_f.index) / total * entropy(list(sub_c.values()))
            if gain > best_attr[0]:
                best_attr = (gain, attr, subs)
        if cls._should_stop(total, classes, best_attr[2], alpha):
            return -1, None, None
        return best_attr

    @staticmethod
    def _should_stop(total, classes, subs, alpha):
        cnames = sorted(classes)
        observed = []
        expected = []
        for val, (sub_features, sub_count) in subs.items():
            sub_len = len(sub_features.index)
            observed.append([sub_count[k] for k in cnames])
            expected.append([classes[k] * sub_len / total for k in cnames])
        chi, pv = chisquare(observed, expected, axis=None)
        return pv > alpha

    def _subtree(self, val):
        sub_tree = Tree(self)
        self.children[val] = sub_tree
        return sub_tree

    def _induce(self, features, label, alpha, classes=None, used_attrs=None):
        if classes is None:
            classes = Counter(features[label])
        if used_attrs is None:
            used_attrs = {label}
        self.label = classes.most_common(1)[0][0]
        if len(classes) == 1 or features.shape[1] == 0:
            print('Stopped with the most common class: {}'
                  .format(classes.most_common()))
        else:
            gain, self.attribute, subs = self._best_attr(used_attrs,
                                                         features, label, classes, alpha)
            if self.attribute is None:
                print('Stopped by Chi2 with the most common class: {}'
                      .format(classes.most_common()))
            else:
                used_attrs = used_attrs | {self.attribute}
                print('Splitting at "{}" with gain {}'.format(self.attribute, gain))
                for val, (sub_features, sub_classes) in subs.items():
                    self._subtree(val)._induce(sub_features, label,
                                               alpha, sub_classes, used_attrs)

    @classmethod
    def induce(cls, features, label, alpha):
        tree = cls()
        tree._induce(features, label, alpha)
        return tree

    def evaluate(self, row):
        if not self.children:
            return self.label
        val = row[self.attribute]
        child = self.children.get(val)
        if child:
            return child.evaluate(row)
        return self.label

    def error_rate(self, features, label):
        labels = features[label]
        total = len(features.index)
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


    # import sys
    # from cProfile import Profile

    def read_data(basename, names=None):
        return pd.read_csv('../data/clickstream/{}.csv'.format(basename),
                           sep=' ', names=names)


    names = read_data('featnames', ['name'])
    trainfeat = read_data('trainfeat', names.name)
    trainfeat['label'] = read_data('trainlabs', ['result']).result
    testfeat = read_data('testfeat', names.name)
    testfeat['label'] = read_data('testlabs', ['result']).result

    savefile = 'clickstream-100.json'
    if os.path.isfile(savefile):
        t = Tree.load(savefile)
    else:
        # prof = Profile()
        # prof.enable()
        t = Tree.induce(trainfeat, 'label', alpha=1)
        # prof.disable()
        # prof.dump_stats('induce.prof')
        t.save(savefile)
    print('Train error rate: {}'.format(t.error_rate(trainfeat, 'label')))
    print('Test error rate: {}'.format(t.error_rate(testfeat, 'label')))

# 1
# Train error rate: 0.0512
# Test error rate: 0.3366
# 0.5
# Train error rate: 0.180325
# Test error rate: 0.27168
# 0.2
# Train error rate: 0.1862
# Test error rate: 0.2664
# 0.1
# Train error rate: 0.1871
# Test error rate: 0.26464
# 0.05
# Train error rate: 0.1877
# Test error rate: 0.25376
# 0.01
# Train error rate: 0.188875
# Test error rate: 0.24916
# 0.001
# Train error rate: 0.191525
# Test error rate: 0.24956
