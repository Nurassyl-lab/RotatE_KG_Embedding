import numpy as np
import matplotlib.pyplot as plt
import sys

# Generate sample 1000-dimensional vectors (each feature on the unit circle)
np.random.seed(42)
DATA = np.load("pRotatE_1000_Entity_Embeddings_FB15k.npy")
Relation_DATA = np.load("pRotatE_1000_Relation_Embeddings_FB15k.npy")

embedding_range_constant = 0.026000000536441803
A = DATA[0]
B = DATA[1]

# Normalize angles to [-pi, pi]
A = np.abs(A/(embedding_range_constant/np.pi))
B = np.abs(B/(embedding_range_constant/np.pi))

A = np.rad2deg((A + np.pi) % (2 * np.pi) - np.pi)  # Keeps values in [-pi, pi]
B = np.rad2deg((B + np.pi) % (2 * np.pi) - np.pi)  # Keeps values in [-pi, pi]

# Compute cumulative sum of angles
A_cumsum = np.cumsum(A)[-1]
B_cumsum = np.cumsum(B)[-1]

# Option 1: Keep angles within [-π, π] using modulo
A_cumsum_wrapped = np.mod(A_cumsum + np.pi, 2 * np.pi) - np.pi
B_cumsum_wrapped = np.mod(B_cumsum + np.pi, 2 * np.pi) - np.pi

print(f"Phase in [-pi, pi]\nA:{A_cumsum_wrapped}\nB:{B_cumsum_wrapped}")