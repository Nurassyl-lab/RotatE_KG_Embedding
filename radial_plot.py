import numpy as np
import matplotlib.pyplot as plt

# Generate sample 1000-dimensional vectors (each feature on the unit circle)
np.random.seed(42)
DATA = np.load("pRotatE_1000_Entity_Embeddings_FB15k.npy")

embedding_range_constant = 0.026000000536441803
A = DATA[0]
B = DATA[1]

# Normalize angles to [0, 2π]
A = A/(embedding_range_constant/np.pi)
B = B/(embedding_range_constant/np.pi)

# Create theta values (1000 evenly spaced angles for radial plot)
theta = np.linspace(0, 2 * np.pi, 1000, endpoint=False)

# Convert to unit circle (r=1)
r_A = np.ones_like(A)
r_B = np.ones_like(B)

# Create polar plot
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})

# Plot vectors
ax.scatter(theta, r_A, c='blue', label='Vector A', s=5, alpha=0.6)
ax.scatter(theta, r_B, c='red', label='Vector B', s=5, alpha=0.6)

# Connect corresponding points to show differences
for i in range(0, 1000, 50):  # Sample a subset for clarity
    ax.plot([theta[i], theta[i]], [r_A[i], r_B[i]], color='gray', alpha=0.5, linewidth=0.8)

# Formatting
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_title("Radial Plot of High-Dimensional Vectors")

plt.legend()
plt.savefig("images/radial_plot_pRotatE_1000.png")
plt.close()