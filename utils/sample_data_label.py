import numpy as np

from collections import defaultdict


def calculate_class_distribution(labels):
    # Calculate the distribution of classes
    labels = np.squeeze(labels)
    unique_classes, counts = np.unique(labels, return_counts=True)
    distribution = dict(zip(unique_classes, counts / len(labels)))

    # Store the indices of each class
    class_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        class_indices[label].append(idx)

    return distribution, class_indices


def generate_random_transition_matrix(limiting_probs, random_seed=None):
    np.random.seed(random_seed)
    n = len(limiting_probs)
    P = np.random.rand(n, n)

    # Normalize rows to sum to 1
    P = P / P.sum(axis=1, keepdims=True)

    # Adjust P to ensure that the stationary distribution approximates the limiting probabilities
    for _ in range(
        1000
    ):  # number of iterations can be adjusted for better approximation
        P = P * limiting_probs / P.sum(axis=0)
        P = P / P.sum(axis=1, keepdims=True)

    return P


def generate_class_samples(
    transition_matrix,
    class_distribution,
    class_indices,
    num_samples,
    start_class=None,
    sample_size=1,
    random_seed=None,
):
    np.random.seed(random_seed)
    classes = list(class_distribution.keys())

    # Initialize the start class
    if start_class is None:
        start_class = np.random.choice(classes)

    current_class = start_class
    samples = [current_class for _ in range(sample_size)]
    indices_samples = [
        np.random.choice(class_indices[current_class]) for _ in range(sample_size)
    ]

    for _ in range(num_samples - 1):
        # Find the index of the current class in the classes list
        current_index = classes.index(current_class)
        # Get the probabilities for the next class
        next_class_probs = transition_matrix[current_index]
        # Choose the next class based on the transition probabilities
        next_class = np.random.choice(classes, p=next_class_probs)
        # add samples based on sample size
        for _ in range(sample_size):
            samples.append(next_class)
            indices_samples.append(np.random.choice(class_indices[next_class]))
        # Update the current class
        current_class = next_class

    return samples, indices_samples


def sample_data_label(labels_train, labels_test, sample_size, random_seed=None):
    # calculate the distribution of classes
    distribution_train, class_indices_train = calculate_class_distribution(labels_train)
    distribution_test, class_indices_test = calculate_class_distribution(labels_test)
    # the number of classes in train and test set has to be same
    assert len(distribution_train.keys()) == len(distribution_test.keys())
    # generate transition matrix
    transition_matrix = generate_random_transition_matrix(
        list(distribution_train.values()), random_seed
    )
    # sample train and test data
    samples_train, indices_samples_train = generate_class_samples(
        transition_matrix,
        distribution_train,
        class_indices_train,
        int(len(labels_train) * 100),
        None,
        sample_size,
        random_seed,
    )
    samples_test, indices_samples_test = generate_class_samples(
        transition_matrix,
        distribution_train,
        class_indices_test,
        int(len(labels_test) * 2),
        None,
        sample_size,
        random_seed,
    )
    return indices_samples_train, indices_samples_test
