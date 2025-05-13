import numpy as np
import random

# From the paper
# P = population size, K=number of clusters, N=dimension of feature space
# S = size of the data, X=size of data points
# mu_c = crossover_probability, mu_m = mutation probability


def initialize_population(grey_values, P, K):
    """Initialize population with K random cluster centers (chromosomes)."""
    return [np.random.choice(grey_values, K, replace=False) for _ in range(P)]


def assign_clusters(grey_values, cluster_centers):
    """Assign each pixel to the nearest cluster center."""
    clusters = [[] for _ in cluster_centers]
    for val in grey_values:
        index = np.argmin([abs(val - cluster) for cluster in cluster_centers])
        clusters[index].append(val)

    # clusters will be a list with a list of all pixels assigned to each cluster
    return clusters


def update_cluster_centers(clusters):
    """Update cluster centers to be the mean of assigned values."""
    return [np.mean(cluster) if cluster else 0 for cluster in clusters]


def validity_index(clusters, centers):
    """Compute Turi's validity index."""
    # intra (measures the compactness of the clusters)
    # "The term intra is the average of all the distances between each data point and its cluster centroid"
    intra = np.sum([np.sum((np.array(cluster) - center) ** 2) / len(cluster) if len(cluster) > 0 else 0
                    for cluster, center in zip(clusters, centers)])
    # inter (measures the separation of the clusters)
    # "The term inter is the minimum distances between the cluster centroids"
    inter = np.min([abs(ci - cj) for i, ci in enumerate(centers) for j, cj in enumerate(centers) if i != j] or [1])
    c = 0.1
    y = c * np.random.normal(2, 1) + 1    # y = c * N(2,1) + 1
                                          # "c is a user specified parameter and N(2,1) is a Gaussian Distribution with mean 2 and standard deviation of 1"
    return y * intra / inter


def apply_mutation(centers, mutation_rate=0.1):
    # Randomly mutate cluster centers
    new_centers = centers.copy()
    for i in range(len(centers)):
        if random.random() < mutation_rate:
            delta = random.uniform(-0.1, 0.1)
            new_centers[i] += delta * centers[i] if centers[i] != 0 else delta
            new_centers[i] = np.clip(new_centers[i], 0, 255)
    return new_centers


def crossover(parent1, parent2):
    # Single-point crossover
    point = random.randint(1, len(parent1) - 1)
    child1 = np.concatenate([parent1[:point], parent2[point:]])
    child2 = np.concatenate([parent2[:point], parent1[point:]])
    return child1, child2


def genetic_clustering(grey_values, num_clusters, initial_population, generations=10, crossover_prob=0.8, mutation_prob=0.1):
    # Main GA loop
    population = [p.copy() for p in initial_population]
    population_size = len(population)

    # Evaluate initial best
    best_palette = population[0]
    assigned_pixels_per_cluster = assign_clusters(grey_values, best_palette)
    best_val_index = validity_index(assigned_pixels_per_cluster, best_palette)

    for i in range(generations):
        print(f"Generation {i}")
        fitness_scores = []
        for palette in population:
            clusters = assign_clusters(grey_values, palette)
            updated_centers = update_cluster_centers(clusters)
            val_index = validity_index(clusters, updated_centers)
            fitness_scores.append((val_index, updated_centers))
            if val_index < best_val_index:
                best_val_index = val_index
                best_palette = updated_centers

        # population selection with elitism, we keep the best individuals after mutation, half of the population
        fitness_scores.sort(key=lambda x: x[0])
        selected = [x[1] for x in fitness_scores[:population_size // 2]]

        # getting the offsprings until the new population size will be again the initial size
        new_population = []
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(selected, 2)
            if random.random() < crossover_prob:
                child1, child2 = crossover(np.array(parent1), np.array(parent2))
            else:
                child1, child2 = parent1, parent2
            new_population.append(apply_mutation(child1, mutation_prob))
            new_population.append(apply_mutation(child2, mutation_prob))

        population = new_population[:population_size]

    return best_palette
