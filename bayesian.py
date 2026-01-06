from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from random import Random

from utils import read_sms


def tokenize_sms(message):
    message = message.lower()
    tokens = re.findall(r"[a-z0-9']+", message)
    return tokens


class MultinomialNaiveBayesClassifier:
    def __init__(self, assumed_probability=1):
        self.assumed_probability = assumed_probability

    def fit(self, observations, labels):
        # Get the set of possible classes (e.g. ham, spam)
        self.classes_ = sorted(set(labels))

        # Count how many documents belong to each class
        self.doc_count_ = Counter(labels)

        # Total number of training documents
        self.total_docs_ = len(labels)

        # Initialize vocabulary and word counters per class
        self.vocabulary_ = set()
        self.word_count_ = {c: Counter() for c in self.classes_}

        # Count word occurrences for each class
        for tokens, c in zip(observations, labels):
            for t in tokens:
                self.vocabulary_.add(t)
                self.word_count_[c][t] += 1

        # Compute log prior probabilities log(P(c))
        self.class_log_prior_ = {}
        for c in self.classes_:
            self.class_log_prior_[c] = math.log(
                self.doc_count_[c] / self.total_docs_
            )

        # Use assumed_probability as additive smoothing (Laplace smoothing)
        # To avoid zero probabilities for unseen tokens
        alpha = float(self.assumed_probability)

        # Vocabulary size
        V = len(self.vocabulary_)

        # Initialize structures for likelihoods
        self.feature_log_prob_ = {}
        self.unknown_log_prob_ = {}

        # For each class, compute token probabilities P(w|c)
        for c in self.classes_:
            self.feature_log_prob_[c] = {}

            # Total number of tokens in this class
            total_words_c = sum(self.word_count_[c].values())

            # Common denominator for all tokens of this class
            denom = total_words_c + alpha * V

            # Compute log-probability for each token in the vocabulary
            for w in self.vocabulary_:
                num = self.word_count_[c][w] + alpha
                self.feature_log_prob_[c][w] = math.log(num / denom)

            # Log-probability for unseen tokens (out-of-vocabulary)
            self.unknown_log_prob_[c] = math.log(alpha / denom)

        return self

    def predict(self, observations):
        predictions = []

        # For each tokenized message
        for tokens in observations:
            token_counts = Counter(tokens)  # How many times each token appears in this message

            best_class = None
            best_score = -float("inf")

            # Compute a score for each class
            for c in self.classes_:
                score = self.class_log_prior_[c]  # Start with log P(c)

                # Add log-likelihoods for each token in the message
                for t, k in token_counts.items():
                    if t in self.vocabulary_:
                        score += k * self.feature_log_prob_[c][t]
                    else:
                        # Unseen token -> use the smoothed unknown probability
                        score += k * self.unknown_log_prob_[c]

                # Keep the best scoring class
                if score > best_score:
                    best_score = score
                    best_class = c

            predictions.append(best_class)

        return predictions

    def score(self, data, labels) -> float:
        predicted = self.predict(data)
        correct = sum(
            1 if pred == expected else 0 for pred, expected in zip(predicted, labels)
        )
        return correct / len(data)


###############################################
#                 CLI Code                    #
###############################################


def main(args):
    # Set the random generator
    rng = Random(args.seed)

    # Load the dataset
    messages, labels = read_sms(args.dataset)

    # Tokenize the messages
    tokenized_messages = [tokenize_sms(m) for m in messages]

    # Split the dataset into training and test sets
    # NOTE: consider args.test_ratio and args.seed
    n = len(tokenized_messages)
    indices = list(range(n))
    # Random number generator
    rng.shuffle(indices)

    # Split the dataset into training and test sets according to the specified test ratio
    test_size = int(n * args.test_ratio)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    train_messages = [tokenized_messages[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    test_messages = [tokenized_messages[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]

    # Instantiate the Naive Bayes classifier
    mnb = MultinomialNaiveBayesClassifier(
        assumed_probability=args.assumed_probability
        )

    # Train the classifier using the training data
    mnb.fit(train_messages, train_labels)

    # Predict over the test set
    predictions = mnb.predict(test_messages)

    # Evaluate predictions using accuracy and print the information
    # Count correct predictions
    correct = sum(
        1 for pred, expected in zip(predictions, test_labels) if pred == expected
        )
    # Calc accuracy following the formula
    accuracy = correct / len(test_labels) if len(test_labels) > 0 else 0.0
    print(f"Accuracy: {accuracy:.4f}")




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset", type=str, help="Path to the CSV file containing the dataset."
    )
    parser.add_argument(
        "--assumed_probability",
        type=int,
        default=1,
        help="Value for the 'assumed_probability' parameter.",
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.3, help="Ratio for the test set split."
    )
    parser.add_argument("--seed", type=int, default=123456, help="RNG Seed.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
