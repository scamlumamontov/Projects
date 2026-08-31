from collections import Counter
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np


def find_best_split(feature_vector, target_vector, min_samples_leaf=None):
    X, y = feature_vector, target_vector

    #Sorting by (x, y)
    cur = np.stack([X, y], axis=1)
    sort_idx = cur[:, 0].argsort()
    X, y = X[sort_idx], y[sort_idx]

    #Split only for different values
    split = (X[1:] != X[:-1])

    #Len of parts
    pref = np.arange(1, len(X), 1)
    suf = np.arange(len(X) - 1, 0, -1)

    if min_samples_leaf is not None:
        mask = (pref > min_samples_leaf) & (suf > min_samples_leaf)
        split = mask & split

    left_ones = np.cumsum(y == 1)
    right_ones = (y == 1).sum() - left_ones

    #Proportions
    left_p = left_ones[:-1] / pref
    right_p = right_ones[1:] / suf

    #Gini calculation
    pref_p = pref / len(X)
    pref_gini = pref_p * (1 - left_p ** 2 - (1 - left_p) ** 2)
    suf_gini = (1 - pref_p) * (1 - right_p ** 2 - (1 - right_p) ** 2)
    gini_total = pref_gini + suf_gini

    gini_total = gini_total[split]
    thresholds = (X[1:] + X[:-1])[split] / 2

    if len(gini_total) == 0:
        return None, None, None, None
    
    best_idx = np.argmin(gini_total)

    return thresholds, gini_total, thresholds[best_idx], gini_total[best_idx]


class DecisionTree(BaseEstimator, ClassifierMixin):
    def __init__(self, feature_types, max_depth=10, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self.tree = {}
        self.feature_types = feature_types
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node):
        try:
            if np.all((sub_y == sub_y[0])):
                node["type"] = "terminal"
                node["class"] = sub_y[0]
                return
        except:
            raise Exception(f"{sub_y}")

        if "parent" in node:
            node["depth"] = node["parent"]["depth"] + 1
        else:
            node["depth"] = 1

        if node["depth"] == self.max_depth:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        if self.min_samples_split is not None and self.min_samples_split > len(sub_y):
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(0, len(self.feature_types)):
            feature_type = self.feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1]) ))
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))

                feature_vector = np.array( list(map(lambda x: categories_map[x], sub_X[:, feature])) )
            else:
                raise ValueError
            #Ok

            #raise Exception(f"{feature_vector.shape}, {sub_y.shape}")

            _, _, threshold, gini = find_best_split(feature_vector, sub_y, self.min_samples_leaf)
            if gini is None:
                continue

            if gini_best is None or gini < gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self.feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self.feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError

        node["left_child"], node["right_child"] = {}, {}
        node["left_child"]["parent"], node["right_child"]["parent"] = node, node

        if(len(split) - split.sum() > 0):
            self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"])
        if(split.sum() > 0):
            self._fit_node(sub_X[split], sub_y[split], node["left_child"])
    
    def _predict_node(self, x, node):
        # ╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ

        if node["type"] == "terminal":
            #raise Exception(f"Predicted: {node["class"]}")
            return node["class"]
        feature = node["feature_split"]

        #raise Exception(f"Feature: {feature}")

        if "categories_split" not in node and "threshold" not in node:
            raise ValueError

        if node.get("categories_split") is not None:
            if x[feature] in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        else:
            if x[feature] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        #pass

    def fit(self, X, y):
        self._fit_node(X, y, self.tree)

    def predict(self, X):
        predicted = []
        for x in X:
            val = self._predict_node(x, self.tree)
            predicted.append(val)
        return np.array(predicted)
