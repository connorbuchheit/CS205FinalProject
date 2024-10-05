import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles, make_moons

# Parameters
n_samples = 1000
centers = [(0, 0), (1, 1)]  # Adjust these for different separations

# Generate linearly separable data
X, y = make_blobs(n_samples=n_samples, centers=centers, n_features=2, cluster_std=0.5)

# Generate non-linearly separable data
X_nl, y_nl = make_moons(n_samples=n_samples, noise=0.1)

# Plotting
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
ax[0].set_title('Linearly Separable Data')
ax[1].scatter(X_nl[:, 0], X_nl[:, 1], c=y_nl, cmap='coolwarm', edgecolors='k')
ax[1].set_title('Non-linearly Separable Data')
plt.show()