from __future__ import annotations
from math import log
from dataclasses import dataclass
from typing import Optional
import argparse
from utils import read_csv, split_observations_and_labels
from random import Random
import itertools


def gini(labels) -> float:
    total = len(labels)
    results = _unique_counts(labels)
    imp = 1

    for label, count in results.items():
        prob = count / total
        imp -= prob**2

    return imp


def entropy(labels):
    total = len(labels)
    results = _unique_counts(labels)
    imp = 0

    for label, count in results.items():
        prob = count / total
        imp -= prob * _log2(prob)
    
    return imp


class DecisionTreeClassifier:
    def __init__(self, scoref=gini, beta=0, prune_threshold=0):
        self.scoref = scoref
        self.beta = beta
        self.prune_threshold = prune_threshold

    def fit(self, observations, labels):
        self._iterative_build_tree(observations, labels)
        self._prune_tree()
        return self

    def predict(self, observations):
        labels = []
        for observation in observations:
            leaf = self.tree_.follow_tree(observation)
            label = max(leaf.results.items(), key=lambda x: x[1])[0]
            labels.append(label)
        return labels

    def score(self, data, labels) -> float:
        predicted = self.predict(data)
        correct = sum(
            1 if pred == expected else 0 for pred, expected in zip(predicted, labels)
        )
        return correct / len(data)

    def _iterative_build_tree(self, observations, labels):
        # create an empty root node to fill it afterwards
        self.tree_ = Node(None, None, None, None, None)

        # stack: every element is: node_to_fill, obs, labels
        stack = [(self.tree_, observations, labels)]

        while stack:
            node, obs, labs = stack.pop()

            # base case
            if not obs or not labs:
                # empty leaf
                node.column = None
                node.value = None
                node.results = _unique_counts(labs)
                node.true_branch = None
                node.false_branch = None
                continue
            
            root_imp = self.scoref(labs)
            if root_imp == 0:
                # pure node -> leaf
                node.column = None
                node.value = None
                node.results = _unique_counts(labs)
                node.true_branch = None
                node.false_branch = None
                continue
            
            best_col, best_val = None, None
            best_goodness = 0.0
            best_split = None # (obs1, labs1, obs2, labs2)

            ncols = len(obs[0])

            # try all available questions (col, value)
            for col in range(ncols):
                col_values = _unique_values(obs, col)

                # numeric column: try thresholds
                # categorical column: try all subsets
                if col_values and all(_is_numeric(v) for v in col_values):
                    candidate_values = col_values
                else:
                    candidate_values = _all_nontrivial_subsets(col_values)

                for value in candidate_values:
                    obs1, labs1, obs2, labs2 = _divideset(obs, labs, col, value)

                    # avoid "useless" splits
                    if len(obs1) == 0 or len(obs2) == 0:
                        continue
                    
                    p1 = len(obs1) / len(obs)
                    p2 = len(obs2) / len(obs)

                    goodness = root_imp - p1 * self.scoref(labs1) - p2 * self.scoref(labs2)

                    if goodness > best_goodness:
                        best_goodness = goodness
                        best_col, best_val = col, value
                        best_split = (obs1, labs1, obs2, labs2)

            # if we don't find a better one -> leaf
            if best_split is None or best_goodness < self.beta:
                node.column = None
                node.value = None
                node.results = _unique_counts(labs)
                node.true_branch = None
                node.false_branch = None
                continue
            
            # if we find split
            node.column = best_col
            node.value = best_val
            node.results = None

            obs1, labs1, obs2, labs2 = best_split

            node.true_branch = Node(None, None, None, None, None)
            node.false_branch = Node(None, None, None, None, None)

            # push child nodes in the stack to build them later
            stack.append((node.true_branch, obs1, labs1))
            stack.append((node.false_branch, obs2, labs2))

    def _prune_tree(self):
        # if threshold is <= 0, no pruning
        if self.prune_threshold <= 0 or self.tree_ is None:
            return
        
        def counts_to_labels(counts_dict):
            labels = []
            for lab, cnt in counts_dict.items():
                labels.extend([lab] * cnt)
            return labels
        
        # (node, visited)
        stack = [(self.tree_, False)]

        while stack:
            node, visited = stack.pop()

            if node is None or node.is_leaf():
                continue
            
            if not visited:
                # process children first, then the node
                stack.append((node, True))
                stack.append((node.true_branch, False))
                stack.append((node.false_branch, False))
                continue
            
            # going "up", if both children are leaves, try to prune
            if node.true_branch is None or node.false_branch is None:
                continue
            
            if node.true_branch.is_leaf() and node.false_branch.is_leaf():
                left_counts = node.true_branch.results
                right_counts = node.false_branch.results

                left_labels = counts_to_labels(left_counts)
                right_labels = counts_to_labels(right_counts)

                n_left = len(left_labels)
                n_right = len(right_labels)
                n_total = n_left + n_right

                if n_total == 0:
                    continue
                
                # impurity if we merge
                merged_labels = left_labels + right_labels
                imp_merged = self.scoref(merged_labels)

                # impurity if we keep the split
                imp_children = (n_left / n_total) * self.scoref(left_labels) + (n_right / n_total) * self.scoref(right_labels)

                # improvement of keeping the split vs merging
                improvement = imp_merged - imp_children

                # if the improvement is small, prune
                if improvement < self.prune_threshold:
                    node.column = None
                    node.value = None
                    node.results = _unique_counts(merged_labels)
                    node.true_branch = None
                    node.false_branch = None




@dataclass
class Node:
    column: Optional[int]
    value: Optional[int | float | str | frozenset]
    results: Optional[dict[int | float | str, int]]
    true_branch: Optional[Node]
    false_branch: Optional[Node]

    def is_leaf(self):
        return self.true_branch is None

    @classmethod
    def new_node(cls, column, value, true_branch, false_branch):
        """Create a new instance of this class representing a decision node."""
        return cls(column, value, None, true_branch, false_branch)

    @classmethod
    def new_leaf(cls, labels):
        """Create a new instance of this class representing a leaf."""
        return cls(None, None, _unique_counts(labels), None, None)
    
    def count_nodes(self):
        # return total number of nodes in the tree
        if self is None:
            return 0
        if self.is_leaf():
            return 1
        return 1 + self.true_branch.count_nodes() + self.false_branch.count_nodes()
    
    def max_depth(self):
        # return the maximum depth in the tree
        if self is None:
            return 0
        if self.is_leaf():
            return 1
        return 1 + max(self.true_branch.max_depth(), self.false_branch.max_depth())

    def print_tree(self, indent=""):
        """Prints to stdout a representation of the tree."""
        if self.is_leaf():
            print(self.results)
        else:
            # Print the criteria
            if _is_numeric(self.value):
                print(f"{self.column}: <= {self.value}?")
            else:
                if isinstance(self.value, (set, frozenset)):
                    print(f"{self.column}: in {set(self.value)}?")
                else:
                    print(f"{self.column}: {self.value}?")
            # Print the branches
            print(f"{indent}T->", end="")
            self.true_branch.print_tree(indent + "-")
            print(f"{indent}F->", end="")
            self.false_branch.print_tree(indent + "-")

    def follow_tree(self, observation):
        """
        Traverse the (sub)tree by answering the queries, until a leaf is reached.

        This method returns the leaf that this observation reaches.
        """
        current = self
        while not current.is_leaf():
            query_fn = _get_query_fn(current.column, current.value)
            current = (
                current.true_branch if query_fn(observation) else current.false_branch
            )

        return current


###############################################
#             UTILITY FUNCTIONS               #
###############################################


def _unique_counts(values):
    """Count how many times each value appears in `values`"""
    results = {}
    for value in values:
        if value not in results:
            results[value] = 1
        else:
            results[value] += 1
    return results


def _is_numeric(value):
    """Checks if a value is numeric (i.e. a float or an int)"""
    return isinstance(value, int) or isinstance(value, float)


def _get_query_fn(column, value):
    """
    Create a function that separates observations based on a query.
    The query can be:

    a) categorical: the created function returns true
       iff. the observation has the exact value in the column specified.
    b) continuous: the created function returns true
       iff. the observation has a value smaller or equal than the
       reference one in the column specified.

    Note: consider any column with a numeric value as continuous.
    """
    if _is_numeric(value):
        return lambda prot: prot[column] <= value
    else:
        if isinstance(value, (set, frozenset)):
            return lambda prot: prot[column] in value 
        return lambda prot: prot[column] == value


def _unique_values(table, column_idx):
    """Returns a set of the values in the columns of a table."""
    values = set()
    for row in table:
        values.add(row[column_idx])
    return values

def _all_nontrivial_subsets(values):
    vals = list(values)
    k = len(vals)
    subsets = []

    for r in range(1, (k // 2) + 1):
        for comb in itertools.combinations(vals, r):
            subsets.append(frozenset(comb))
    
    return subsets


def _log2(x):
    return log(x) / log(2)


def _divideset(observations, labels, column, value):
    """
    Divides a set on a specific column.
    Can handle numeric or categorical values
    """
    query_fn = _get_query_fn(column, value)

    observations1, labels1, observations2, labels2 = [], [], [], []

    for row, label in zip(observations, labels):
        if query_fn(row):
            observations1.append(row)
            labels1.append(label)
        else:
            observations2.append(row)
            labels2.append(label)

    return observations1, labels1, observations2, labels2


###############################################
#                 CLI Code                    #
###############################################


def main(args):
    # Set the random generator
    rng = Random(args.seed)

    # Load the dataset
    dataset = read_csv(args.dataset, ignore_first=True)
    observations, labels = split_observations_and_labels(dataset)

    # Split the dataset into training and test sets
    # NOTE: consider args.test_ratio and args.seed
    idxs = list(range(len(observations)))
    rng.shuffle(idxs)

    n_test = int(args.test_ratio * len(observations))
    test_idxs = set(idxs[:n_test])

    train_X, test_X, train_y, test_y = [], [], [], []
    for i, (obs, lab) in enumerate(zip(observations, labels)):
        if i in test_idxs:
            test_X.append(obs)
            test_y.append(lab)
        else:
            train_X.append(obs)
            train_y.append(lab)
    
    if args.scoref == "gini":
        scoref = gini
    else:
        scoref = entropy

    # Instantiate the decision tree classifier
    dec_tree = DecisionTreeClassifier(
        scoref=scoref, beta=args.beta, prune_threshold=args.prune_threshold
    )

    # Train the decision tree using the training data
    dec_tree.fit(train_X, train_y)

    # Print the tree structure
    print("Tree Structure:")
    dec_tree.tree_.print_tree()

    # print the stats:
    print("Tree stats:")
    print("Depth:", dec_tree.tree_.max_depth())
    print("Number of nodes:", dec_tree.tree_.count_nodes())


    # Predict over the test set
    predictions = dec_tree.predict(test_X)

    # Evaluate these predictions using the accuracy score and print the information
    accuracy = dec_tree.score(test_X, test_y)
    print("Accuracy: ", accuracy)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset", type=str, help="Path to the CSV file containing the dataset."
    )
    parser.add_argument(
        "--scoref",
        type=str,
        choices=["gini", "entropy"],
        default="gini",
        help="Impurity measure criterion for the decision tree.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.0,
        help="Value for the 'beta' parameter in the decision tree.",
    )
    parser.add_argument(
        "--prune-threshold", type=float, default=0.0, help="Prune threshold."
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.3, help="Ratio for the test set split."
    )
    parser.add_argument("--seed", type=int, default=123456, help="RNG Seed.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
