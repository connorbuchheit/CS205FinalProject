import matplotlib.pyplot as plt
import numpy as np

def calculate_speedup(data_size, time):
    reference_time = time[0]
    return [reference_time * data_size[0] / t for t in time]

def calculate_efficiency(speedup, data_size):
    return [speed / p for speed, p in zip(speedup, data_size)]

def plot_performance(data_size, speedup, efficiency, ideal_speedup, ideal_efficiency, dataset_size):
    plt.figure(figsize=(10, 5))

    # Speedup plot
    plt.subplot(1, 2, 1)
    plt.plot(data_size, speedup, marker='o', color='b', linestyle='-', label='Actual Speedup')
    plt.plot(data_size, ideal_speedup, marker='', color='g', linestyle='--', label='Ideal Speedup')
    plt.title(f'Speedup vs. Number of Machines Data={dataset_size}k')
    plt.xlabel('Number of Machines')
    plt.ylabel('Speedup')
    plt.legend()
    plt.grid(True)

    # Efficiency plot
    plt.subplot(1, 2, 2)
    plt.plot(data_size, efficiency, marker='o', color='r', linestyle='-', label='Actual Efficiency')
    plt.plot(data_size, ideal_efficiency, marker='', color='m', linestyle='--', label='Ideal Efficiency')
    plt.title(f'Efficiency vs. Number of Machines Data={dataset_size}k')
    plt.xlabel('Number of Machines')
    plt.ylabel('Efficiency')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"Strong_Scale_Analysis Data={dataset_size}k")
    plt.show()

def calculate_percentage(communication, computation):
    total = communication + computation
    communication_percentage = (communication / total) * 100
    computation_percentage = (computation / total) * 100
    return communication_percentage, computation_percentage

def plot_percentage(machines, communication_percentage, computation_percentage, dataset_size):
    bar_width = 0.35
    index = np.arange(len(machines))

    fig, ax = plt.subplots()
    bar1 = ax.bar(index, communication_percentage, bar_width, label='Communication')
    bar2 = ax.bar(index + bar_width, computation_percentage, bar_width, label='Computation')

    # Adding labels, title, and custom x-axis tick labels
    ax.set_xlabel('# of Machines')
    ax.set_ylabel('Percentage')
    ax.set_title(f'Communication and Computation as Percentage of Total Time Data={dataset_size}k')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(machines)
    ax.legend()

    # Display the plot
    plt.show()
    plt.savefig(f"Overhead Analysis Data={dataset_size}k")

def weak_scale_analyis():


    # Given data
    machines = np.array([1, 2, 4, 8, 16, 32])
    work_size = np.array([1600, 3200, 6400, 12800, 25600, 51200])
    avg_time = np.array([455.4573333, 492.9096667, 514.399, 661.369, 648.583, 904.532])

    # Calculate speedup and efficiency
    speedup = avg_time[0] * machines / avg_time
    efficiency = speedup / machines

    # Plotting the graph
    plt.figure(figsize=(10,6))

    plt.subplot(2, 1, 1)
    plt.plot(machines, speedup, 'o-', label='Speed Up')
    plt.xlabel('Number of Machines')
    plt.ylabel('Speed Up')
    plt.title('Weak Scaling Analysis')
    plt.grid(False)
    plt.xticks(machines)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(machines, efficiency, 'o-', label='Efficiency')
    plt.xlabel('Number of Machines')
    plt.ylabel('Efficiency')
    plt.grid(False)
    plt.xticks(machines)
    plt.legend()

    plt.tight_layout()
    plt.show()
    plt.savefig("Weak_Scale_Analysis")


def main():
    # Given data
    # Data for 10k
    data_size_10k = [5, 10, 15, 20, 25, 30, 35, 40, 45]
    time_10k = [244.877, 70.3378, 48.1078, 38.4354, 34.9092, 33.2573, 32.223, 31.3362, 28.6196]
    ideal_speedup_10k = [5, 10, 15, 20, 25, 30, 35, 40, 45]
    ideal_efficiency_10k = [1.0] * len(data_size_10k)

    # Data for 50k
    data_size_50k = [10, 15, 20, 25, 30, 35, 40, 45]
    time_50k = [2001.32, 1604.48, 1127.15, 960.227, 806.605, 749.918, 719.774, 702.064]
    ideal_speedup_50k = [10, 15, 20, 25, 30, 35, 40, 45]
    ideal_efficiency_50k = [1.0] * len(data_size_50k)

    # Calculate speedup and efficiency
    speedup_10k = calculate_speedup(data_size_10k, time_10k)
    efficiency_10k = calculate_efficiency(speedup_10k, data_size_10k)

    speedup_50k = calculate_speedup(data_size_50k, time_50k)
    efficiency_50k = calculate_efficiency(speedup_50k, data_size_50k)

    # Plotting
    plot_performance(data_size_10k, speedup_10k, efficiency_10k, ideal_speedup_10k, ideal_efficiency_10k, 10)
    plot_performance(data_size_50k, speedup_50k, efficiency_50k, ideal_speedup_50k, ideal_efficiency_50k, 50)

    weak_scale_analyis()

    # Data for 10k
    machines_10k = [10, 20, 30, 40, 48]
    communication_10k = np.array([0.137346, 0.0746704, 0.028086, 0.0105262, 0.0497279])
    computation_10k = np.array([20.893, 2.78609, 0.891897, 0.440084, 0.285863])

    communication_percentage_10k, computation_percentage_10k = calculate_percentage(communication_10k, computation_10k)

    # Data for 50k
    machines_50k = [20, 30, 40, 48]
    communication_50k = np.array([2.37357, 0.518076, 0.0771924, 0.0661727])
    computation_50k = np.array([118.539, 30.8502, 12.6438, 7.28403])

    communication_percentage_50k, computation_percentage_50k = calculate_percentage(communication_50k, computation_50k)

    # Plotting
    plot_percentage(machines_10k, communication_percentage_10k, computation_percentage_10k, 10)
    plot_percentage(machines_50k, communication_percentage_50k, computation_percentage_50k, 50)

if __name__ == "__main__":
    main()
