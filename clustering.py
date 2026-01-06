import argparse
from random import Random
from utils import read_csv
import math


class KMeans:
    def __init__(
        self,
        k: int = 4,
        distance: str = "euclidean",
        rng=Random(123456),
        n_restarts: int = 10,
    ):
        self.k = k
        self.distance = distance
        self.rng = rng
        self.n_restarts = n_restarts

    def _squared_euclidean(self, a, b) -> float:
        return sum((x - y) ** 2 for x, y in zip(a, b))

    def _distance(self, a, b) -> float:
        d2 = self._squared_euclidean(a, b)
        if self.distance == "squared-euclidean":
            return d2
        return math.sqrt(d2)

    def fit(self, observations):
        n = len(observations)
        if n == 0:
            raise ValueError("Empty dataset.")
        if self.k > n:
            raise ValueError("k cannot be greater than number of observations.")

        best_total_distance = float("inf")
        best_centroids = None
        best_assignments = None
        best_distances = None

        max_iter = 100

        for _ in range(self.n_restarts):
            indices = list(range(n))
            self.rng.shuffle(indices)
            self.centroids_ = [observations[i][:] for i in indices[: self.k]]

            prev_assignments = None

            for _it in range(max_iter):
                self.X_assignments_ = []
                self.distances_ = []

                for x in observations:
                    best_centroid = None
                    best_distance = float("inf")

                    for idx, c in enumerate(self.centroids_):
                        d = self._distance(x, c)
                        if d < best_distance:
                            best_distance = d
                            best_centroid = idx

                    self.X_assignments_.append(best_centroid)
                    self.distances_.append(best_distance)

                # Convergence check
                if prev_assignments == self.X_assignments_:
                    break
                prev_assignments = self.X_assignments_[:]

                dim = len(observations[0])
                new_centroids = [[0.0] * dim for _ in range(self.k)]
                counts = [0] * self.k

                for x, cluster_id in zip(observations, self.X_assignments_):
                    counts[cluster_id] += 1
                    for j in range(dim):
                        new_centroids[cluster_id][j] += x[j]

                for c in range(self.k):
                    if counts[c] == 0:
                        # Empty cluster: reinitialize randomly
                        new_centroids[c] = observations[self.rng.randrange(n)][:] 
                    else:
                        for j in range(dim):
                            new_centroids[c][j] /= counts[c]

                self.centroids_ = new_centroids

            total_distance = sum(self.distances_)
            if total_distance < best_total_distance:
                best_total_distance = total_distance
                best_centroids = [c[:] for c in self.centroids_]
                best_assignments = self.X_assignments_[:]
                best_distances = self.distances_[:]

        # Keep the best solution
        self.centroids_ = best_centroids
        self.X_assignments_ = best_assignments
        self.distances_ = best_distances

        return self

###############################################
#                 CLI Code                    #
###############################################


def main(args):
    # Set the random generator
    rng = Random(args.seed)

    # Load the dataset (ignore first column if it's an identifier)
    dataset = read_csv(args.dataset, ignore_first=True, ignore_first_column=True)

    # Instantiate KMeans
    kmeans = KMeans(
        k=args.k,
        distance=args.distance,
        rng=rng,
        n_restarts=args.n_restarts,
    )


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
    parser.add_argument(
        "--n-restarts",
        type=int,
        default=10,
        help="Number of random restarts for KMeans.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)