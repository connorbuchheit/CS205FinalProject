import numpy as np
import matplotlib.pyplot as plt

# 
peak_flops = 18 * 3.0 * 16 * 1000  # GFLOPS
memory_bandwidth = 128  # GB/s

# Create a range of operational intensities
operational_intensities = np.logspace(-4, 4, base=10, num=1000)  # from 0.0001 to 10000 FLOPs/Byte

# Compute the performance limits
memory_bound_performance = memory_bandwidth * operational_intensities
compute_bound_performance = np.array([peak_flops] * len(operational_intensities))

# Plot the roofline model
plt.figure(figsize=(10, 6))
plt.loglog(operational_intensities, memory_bound_performance, label='Memory Bound', linestyle='--', color='blue')
plt.loglog(operational_intensities, compute_bound_performance, label='Compute Bound', linestyle='--', color='red')

plt.scatter([161.50], [0.34], color='blue', s=100, edgecolor='black', label='Parallel')
plt.scatter([150], [0.34/8], color='red', s=100, edgecolor='black', label='Serialized')

plt.xlabel('Operational Intensity (FLOPs/Byte)')
plt.ylabel('Performance (GFLOPS/s)')
plt.title('Roofline Model')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.savefig("roofline.png")