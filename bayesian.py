from __future__ import annotations
import argparse
from utils import read_sms, split_observations_and_labels
from random import Random
import re


def tokenize_sms(message):
    """
    Convierte un mensaje SMS en una lista de tokens:
    - pasa a minúsculas
    - elimina puntuación
    - separa en palabras alfanuméricas
    """
    message = message.lower()
    tokens = re.findall(r"[a-z0-9']+", message)
    return tokens


class MultinomialNaiveBayesClassifier:
    def __init__(self, assumed_probability=1):
        self.assumed_probability = assumed_probability

    def fit(self, observations, labels):
        """YOUR CODE HERE"""
        return self

    def predict(self, observations):
        """YOUR CODE HERE"""
        raise NotImplementedError("TODO")

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
    #random number generator
    rng.shuffle(indices)

    test_size = int(n * args.test_ratio)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    train_messages = [tokenized_messages[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    test_messages = [tokenized_messages[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]

    # Instantiate the Naive Bayes classifier
    mnb = MultinomialNaiveBayesClassifier(assumed_probability=args.assumed_probability)

    # Train the classifier using the training data
    mnb.fit(train_messages, train_labels)

    # Predict over the test set
    predictions = mnb.predict(test_messages)

    # Evaluate predictions using accuracy and print the information
    #Count correct predictions
    correct = sum(1 for pred, expected in zip(predictions, test_labels) if pred == expected)
    #Calc accuracy following the formula
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
