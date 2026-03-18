# ================= CUSTOM STRONG MODEL =================

class BetterTree:
    def __init__(self, depth=2):
        self.depth = depth
        self.split_feature = None
        self.split_value = None
        self.left = None
        self.right = None
        self.value = None

    def fit(self, X, y, depth=0):
        if depth == self.depth or len(X) < 5:
            self.value = sum(y) / len(y)
            return

        best_error = float('inf')

        for feature in range(len(X[0])):
            values = [row[feature] for row in X]

            for val in values:
                left_X, left_y, right_X, right_y = [], [], [], []

                for i in range(len(X)):
                    if X[i][feature] <= val:
                        left_X.append(X[i])
                        left_y.append(y[i])
                    else:
                        right_X.append(X[i])
                        right_y.append(y[i])

                if len(left_X) == 0 or len(right_X) == 0:
                    continue

                left_mean = sum(left_y) / len(left_y)
                right_mean = sum(right_y) / len(right_y)

                error = sum((v - left_mean) ** 2 for v in left_y) + \
                        sum((v - right_mean) ** 2 for v in right_y)

                if error < best_error:
                    best_error = error
                    self.split_feature = feature
                    self.split_value = val
                    best_left_X, best_left_y = left_X, left_y
                    best_right_X, best_right_y = right_X, right_y

        self.left = BetterTree(self.depth)
        self.right = BetterTree(self.depth)

        self.left.fit(best_left_X, best_left_y, depth+1)
        self.right.fit(best_right_X, best_right_y, depth+1)

    def predict_row(self, row):
        if self.value is not None:
            return self.value

        if row[self.split_feature] <= self.split_value:
            return self.left.predict_row(row)
        else:
            return self.right.predict_row(row)

    def predict(self, X):
        return [self.predict_row(row) for row in X]


class BetterXGBoost:
    def __init__(self, n_estimators=20, learning_rate=0.2):
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.trees = []

    def fit(self, X, y):
        preds = [sum(y)/len(y)] * len(y)

        for _ in range(self.n_estimators):
            residuals = [y[i] - preds[i] for i in range(len(y))]

            tree = BetterTree(depth=2)
            tree.fit(X, residuals)

            update = tree.predict(X)

            preds = [preds[i] + self.lr * update[i] for i in range(len(y))]
            self.trees.append(tree)

    def predict(self, X):
        preds = [0] * len(X)

        for tree in self.trees:
            update = tree.predict(X)
            preds = [preds[i] + self.lr * update[i] for i in range(len(X))]

        return preds