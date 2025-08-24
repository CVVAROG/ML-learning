import numpy as np
from collections import Counter

def entropy(y):
    counts = np.bincount(y)
    probabilities = counts / len(y)
    return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

def split(X_column, threshold):
    left = lambda row: row <= threshold
    right = lambda row: row > threshold
    return left, right

def information_gain(y, left_mask, right_mask):
    H_before = entropy(y)
    n = len(y)
    n_left, n_right = np.sum(left_mask), np.sum(right_mask)
    if n_left == 0 or n_right == 0:
        return 0
        
    H_left = entropy(y[left_mask])
    H_right = entropy(y[right_mask])
    H_after = (n_left / n) * H_left + (n_right / n) * H_right
    return H_before - H_after

class DecisionTree:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(np.array(X), np.array(y), depth=0)

    def _build_tree(self, X, y, depth):
        num_samples, num_features = X.shape
        if depth >= self.max_depth or len(set(y)) == 1:
            return Counter(y).most_common(1)[0][0]

        best_gain = 0
        best_feature = None
        best_threshold = None
        best_split = None

        for feature in range(num_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = X[:, feature] > threshold
                gain = information_gain(y, left_mask, right_mask)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
                    best_split = (left_mask, right_mask)

        if best_gain == 0:
            return Counter(y).most_common(1)[0][0]

        left_subtree = self._build_tree(X[best_split[0]], y[best_split[0]], depth + 1)
        right_subtree = self._build_tree(X[best_split[1]], y[best_split[1]], depth + 1)

        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_subtree,
            'right': right_subtree
        }
    def _predict_one(self, x, node):
        if not isinstance(node, dict):
            return node
        if x[node['feature']] <= node['threshold']:
            return self._predict_one(x, node['left'])
        else:
            return self._predict_one(x, node['right'])

    def predict(self, X):
        return [self._predict_one(x, self.tree) for x in X]
def print_tree(node, depth=0):
    indent = "  " * depth
    if isinstance(node, dict):
        print(f"{indent}X[{node['feature']}] <= {node['threshold']:.2f}?")
        print(f"{indent}--> True:")
        print_tree(node['left'], depth + 1)
        print(f"{indent}--> False:")
        print_tree(node['right'], depth + 1)
    else:
        print(f"{indent}Predict: {node}")
X = np.array([[2,3],[1,5],[6,2],[7,3],[8,6]])
y = np.array([0,0,1,1,1])

dt = DecisionTree(max_depth=3)
dt.fit(X, y)
print_tree(dt.tree)
