import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def get_args():
    parser = argparse.ArgumentParser(description="Compare embeddings")
    parser.add_argument("--entity_embedding", type=str, default="pRotatE_1000_Entity_Embeddings_FB15k.npy", help="Entity embedding file")
    return parser.parse_args()

if __name__ == "__main__":
    print("PCA")
    args = get_args()
    entity_embeddings = np.load(args.entity_embedding)

    # Perform PCA
    pca = PCA()
    pca.fit(entity_embeddings)

    # Calculate cumulative explained variance
    cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)

    # Plot cumulative explained variance
    plt.plot(cumulative_explained_variance)
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("Cumulative Explained Variance vs. Number of Components")
    plt.grid(True)
    filename = args.entity_embedding.split('.')[0]
    plt.savefig(f"images/pca_{filename}.png")
    plt.close()

    # Determine the number of features that explain most of the data (e.g., 95%)
    desired_variance = 0.95
    num_components = np.argmax(cumulative_explained_variance >= desired_variance) + 1

    print(f"Number of components to explain {desired_variance * 100}% of the variance: {num_components}")

    # Determine the number of features that explain most of the data (e.g., 90%)
    desired_variance_90 = 0.90
    num_components_90 = np.argmax(cumulative_explained_variance >= desired_variance_90) + 1

    print(f"Number of components to explain {desired_variance_90 * 100}% of the variance: {num_components_90}")