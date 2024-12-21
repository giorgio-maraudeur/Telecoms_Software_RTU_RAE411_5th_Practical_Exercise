# Task 2: SDN Traffic classification with Decision Tree 
import pandas as pd
import numpy as np

class DecisionTreeClassifierManual:
    def __init__(self, max_depth=None, algorithm='CART'):
        """
        Parameters:
        - max_depth: Maximum depth of the tree. If None, the tree will grow until all leaves are pure.
        - algorithm: 'CART' (Gini Impurity) or 'ID3' (Information Gain)
        """
        self.max_depth = max_depth
        self.algorithm = algorithm
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])
    
    def _gini(self, y):
        """Calculate Gini Impurity"""
        classes, counts = np.unique(y, return_counts=True)
        impurity = 1 - np.sum((count / len(y))**2 for count in counts) 
        return impurity

    def _entropy(self,y):
        """Calculate Entropy"""
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities + 1e-9))
    
    def _information_gain(self, y, y_left, y_right):
        """Calculate Information Gain"""
        parent_entropy = self._entropy(y)
        left_entropy = self._entropy(y_left)
        right_entropy = self._entropy(y_right)
        weighted_entropy = (len(y_left) / len(y)) * left_entropy + (len(y_right) / len(y)) * right_entropy
        return parent_entropy - weighted_entropy
    
    def _best_split(self, X, y):
        """Find the best split based on the chosen algorithm (CART or ID3)"""
        best_feature=None
        best_threshold=None
        best_metric = -float("inf") if self.algorithm =="ID3" else float("inf")
        n_features = X.shape[1]

        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                _, _, y_left, y_right = self._split(X, y, feature_index, threshold)
                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                if self.algorithm == 'CART':
                    gini_left = self._gini(y_left)
                    gini_right = self._gini(y_right)
                    weighted_gini = (len(y_left) / len(y)) * gini_left + (len(y_right) / len(y)) * gini_right
                    metric = weighted_gini
                    if metric < best_metric:
                        best_metric = metric
                        best_feature = feature_index
                        best_threshold = threshold

                    elif self.algorithm == 'ID3':
                        gain = self._information_gain(y, y_left, y_right)
                        if gain > best_metric:
                            best_metric = gain
                            best_feature = feature_index
                            best_threshold = threshold

        return best_feature, best_threshold
    
    def _split(self, X, y, feature_index, threshold):
        """Split the dataset into left and right branches"""
        left_indices = X[:, feature_index] < threshold
        right_indices = ~left_indices
        return X[left_indices], X[right_indices], y[left_indices], y[right_indices]
    
    def _build_tree(self, X, y, depth):
        """Recursively build the decision tree"""
        if len(np.unique(y)) == 1 or len(y) == 0 or (self.max_depth is not None and depth >= self.max_depth):
            return {"type": "leaf", "class": np.bincount(y).argmax()}
        
        feature, threshold = self._best_split(X, y)
        if feature is None or threshold is None:
            return {"type": "leaf", "class": np.bincount(y).argmax()}
        
        X_left, X_right, y_left, y_right = self._split(X, y, feature, threshold)

        return {
            "type": "node",
            "feature_index": feature,
            "threshold": threshold,
            "left": self._build_tree(X_left, y_left, depth + 1),
            "right": self._build_tree(X_right, y_right, depth + 1)
        }

    def _traverse_tree(self, x, node):
        """Traverse the tree to make predictions"""
        if node["type"] == "leaf":
            return node["class"]
        
        if x[node["feature_index"]] < node["threshold"]:
            return self._traverse_tree(x, node["left"])
        else:
            return self._traverse_tree(x, node["right"])
        

# Load the dataset
SDN_data_path = "C:\\Users\\NZUZI MANIEMA\\Documents\\AERO 4\\Riga Semestre 7\\Cours\\Telecomunications Software (RAE 411)\\Lab Works\\Mes cuts\\Fifth Practical Exercise\\SDN_traffic.csv"
dataset = pd.read_csv(SDN_data_path)

print("Dataset loaded successfully\n")
print("The first 5 rows of the dataset are:")
print(dataset.head()) 
print(dataset.info())# Check for missing values
print(dataset.describe()) # Check for outliers
print(dataset.duplicated()) # Check for duplicates
print("The columns in the dataset are:")
print(dataset.columns)


X = dataset [[ 'forward_bps_var',
              "tp_src","tp_dst", "nw_proto",
              "forward_pc", "forward_bc", "forward_pl",
              "forward_piat", "forward_pps", "forward_bps", "forward_pl_mean",
              "forward_piat_mean",   "forward_pps_mean", "forward_bps_mean", "forward_pl_var", "forward_piat_var",
              "forward_pps_var", "forward_pl_q1",   "forward_pl_q3",
              "forward_piat_q1", "forward_piat_q3", "forward_pl_max","forward_pl_min",
              "forward_piat_max", "forward_piat_min", "forward_pps_max", "forward_pps_min",
              "forward_bps_max", "forward_bps_min", "forward_duration", "forward_size_packets",
              "forward_size_bytes", "reverse_pc", "reverse_bc", "reverse_pl", "reverse_piat",
              "reverse_pps", "reverse_bps", "reverse_pl_mean", "reverse_piat_mean", "reverse_pps_mean",
              "reverse_bps_mean", "reverse_pl_var", "reverse_piat_var", "reverse_pps_var", "reverse_bps_var",
              "reverse_pl_q1", "reverse_pl_q3", "reverse_piat_q1", "reverse_piat_q3", "reverse_pl_max",
              "reverse_pl_min", "reverse_piat_max", "reverse_piat_min", "reverse_pps_max", "reverse_pps_min",
              "reverse_bps_max", "reverse_bps_min", "reverse_duration", "reverse_size_packets", "reverse_size_bytes"]]

# Replace the missing values in the forward_bps_var column
X.loc[1877, 'forward_bps_var'] = float(11968065203349)
X.loc[1931, 'forward_bps_var'] = float(12880593804833)
X.loc[2070, 'forward_bps_var'] = float(9022747730895)
X.loc[2381, 'forward_bps_var'] = float(39987497172945)
X.loc[2562, 'forward_bps_var'] = float(663300742992)
X.loc[2567, 'forward_bps_var'] = float(37770223877794)
X.loc[2586, 'forward_bps_var'] = float(97227875083751)
X.loc[2754, 'forward_bps_var'] = float(18709751403737)
X.loc[2765, 'forward_bps_var'] = float(33969277035759)
X.loc[2904, 'forward_bps_var'] = float(39204786962856)
X.loc[3044, 'forward_bps_var'] = float(9169996063653)
X.loc[3349, 'forward_bps_var'] = float(37123283690575)
X.loc[3507, 'forward_bps_var'] = float(61019064590464)
X.loc[3610, 'forward_bps_var'] = float(46049620984072)
X.loc[3717, 'forward_bps_var'] = float(97158873841506)
X.loc[3845, 'forward_bps_var'] = float(11968065203349)
X.loc[3868, 'forward_bps_var'] = float(85874278395372)

print("The values have been replaced successfully\n")

#X = X.drop([1877, 1931, 2070, 2381, 2562, 2567, 2586, 2754, 2765, 2904, 3044, 3349, 3507, 3610, 3717, 3845, 3868], axis=0)
X["forward_bps_var"] = pd.to_numeric(X["forward_bps_var"]) # Convert the column to numeric
print(X.info())

# Convert the categorical column to numerical
Y = dataset[["category"]]
Y = Y.to_numpy()
Y = Y.ravel()
labels, uniques = pd.factorize(Y)
Y = labels
Y = Y.ravel()

# Normalize the data
import scipy.stats as stats
X = stats.zscore(X)
X = np.nan_to_num(X) # Replace NaN values with 0

#Train decision tree classifier
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42, test_size=0.3)


clf_cart = DecisionTreeClassifierManual(max_depth=6, algorithm='ID3')
clf_cart.fit(X_train, Y_train)
print("CART predictions:", clf_cart.predict(X_train))

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, classification_report
from sklearn.model_selection import cross_val_score, KFold
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, Y_train)
accuracy = clf.score(X_test, Y_test)
print("ID3 Predicitons accuracy:", accuracy)


# Evaluate metrics performance

cv = KFold(n_splits=10, random_state=0, shuffle=True)   
accuracy = clf.score(X_test, Y_test)
KFold10_accuracy = cross_val_score(clf, X_train, Y_train, scoring='accuracy', cv=cv, n_jobs=-1)
print(KFold10_accuracy.mean())
predict = clf.predict(X_test)
cm = confusion_matrix(Y_test, predict)
precision = precision_score(Y_test, predict, average='weighted', labels=np.unique(predict))
recall = recall_score(Y_test, predict, average='weighted', labels=np.unique(predict))
f1scoreMacro = f1_score(Y_test, predict, average='macro', labels=np.unique(predict))
print(classification_report(Y_test, predict, target_names=uniques))

# Find out the most important 10 features
importance = clf.feature_importances_
important_features_dict = {}
for idx, val in enumerate(importance):
    important_features_dict[idx] = val
important_features_list = sorted(important_features_dict, key=important_features_dict.get, reverse=True)
print(f'10 most important features: {important_features_list[:10]}')


#plot decision tree and confusion matrix

fn=['forward_bps_var',
              "tp_src","tp_dst", "nw_proto",
              "forward_pc", "forward_bc", "forward_pl",
              "forward_piat", "forward_pps", "forward_bps", "forward_pl_mean",
              "forward_piat_mean",   "forward_pps_mean", "forward_bps_mean", "forward_pl_var", "forward_piat_var",
              "forward_pps_var", "forward_pl_q1",   "forward_pl_q3",
              "forward_piat_q1", "forward_piat_q3", "forward_pl_max","forward_pl_min",
              "forward_piat_max", "forward_piat_min", "forward_pps_max", "forward_pps_min",
              "forward_bps_max", "forward_bps_min", "forward_duration", "forward_size_packets",
              "forward_size_bytes", "reverse_pc", "reverse_bc", "reverse_pl", "reverse_piat",
              "reverse_pps", "reverse_bps", "reverse_pl_mean", "reverse_piat_mean", "reverse_pps_mean",
              "reverse_bps_mean", "reverse_pl_var", "reverse_piat_var", "reverse_pps_var", "reverse_bps_var",
              "reverse_pl_q1", "reverse_pl_q3", "reverse_piat_q1", "reverse_piat_q3", "reverse_pl_max",
              "reverse_pl_min", "reverse_piat_max", "reverse_piat_min", "reverse_pps_max", "reverse_pps_min",
              "reverse_bps_max", "reverse_bps_min", "reverse_duration", "reverse_size_packets", "reverse_size_bytes"]

import matplotlib.pyplot as plt
import seaborn as sn
la = ['WWW', 'DNS', 'FTP', 'ICMP', 'P2P', 'VOIP']
plt.figure(1, dpi =300)
fig = tree.plot_tree(clf, filled = True, feature_names = fn, class_names = la)
plt.title("Decision Tree trained on all features")
plt.show()

labels = uniques
plt.figure(2, figsize=(7,4))
plt.title("Confusion Matrix", fontsize=10)

# Normalize the confusion matrix
cmnew = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sn.heatmap(cmnew, annot=True, cmap='YlGnBu', fmt =".2f", xticklabels=labels, yticklabels=labels)
#plot the heatmap
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

