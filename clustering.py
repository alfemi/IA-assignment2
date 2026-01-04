import argparse
from random import Random
from utils import read_csv
import math


class KMeans:
    def __init__(self, k: int = 4, distance: str = "euclidean", rng=Random(123456)):
        self.k = k
        self.distance = distance
        self.rng = rng

    def fit(self, observations):
        n = len(observations)
        if n == 0:
            raise ValueError("Empty dataset.")
        if self.k > n:
            raise ValueError("k cannot be greater than number of observations.")

        indices = list(range(n))
        self.rng.shuffle(indices)
        self.centroids_ = [observations[i][:] for i in indices[: self.k]]

        # Will be filled later
        self.distances_ = []
        self.X_assignments_ = []
        return self
    
    def _squared_euclidean(self, a, b) -> float:
        return sum((x - y) ** 2 for x, y in zip(a, b))

    def _distance(self, a, b) -> float:
        d2 = self._squared_euclidean(a, b)
        if self.distance == "squared-euclidean":
            return d2
        return math.sqrt(d2)

###############################################
#                 CLI Code                    #
###############################################


def main(args):
    # Set the random generator
    rng = Random(args.seed)

    # Load the dataset (ignore first column if it's an identifier)
    dataset = read_csv(args.dataset, ignore_first_column=True)

    # Instantiate KMeans
    kmeans = KMeans(k=args.k, distance=args.distance, rng=rng)

    # Train the clustering model
    kmeans.fit(dataset)

    # Print some metrics
    print("Distances:", kmeans.distances_)
    print("Sum of distances:", sum(kmeans.distances_))
    print("Centroid positions:", kmeans.centroids_)
    print("Centroids assignments:", kmeans.X_assignments_)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset", type=str, help="Path to the CSV file containing the dataset."
    )
    parser.add_argument(
        "--k", type=int, default=4, help="Value for the 'k' parameter of KMeans."
    )
    parser.add_argument(
        "--distance",
        type=str,
        choices=["euclidean", "squared-euclidean"],
        default="euclidean",
        help="Distance metric used by KMeans.",
    )
    parser.add_argument("--seed", type=int, default=123456, help="RNG Seed.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
