import numpy as np
import imageio
from matplotlib import pyplot as plt
import sys
import os

def mykmeans(pixels, K):
    """
    Your goal of this assignment is implementing your own K-means.

    Input:
        pixels: data set. Each row contains one data point. For image
        dataset, it contains 3 columns, each column corresponding to Red,
        Green, and Blue component.

        K: the number of desired clusters. Too high value of K may result in
        empty cluster error. Then, you need to reduce it.

    Output:
        clusters: the class assignment of each data point in pixels. The
        assignment should be 0, 1, 2, etc. For K = 5, for example, each cell
        of class should be either 0, 1, 2, 3, or 4. The output should be a
        column vector with size(pixels, 1) elements.

        centroid: the location of K centroids in your result. With images,
        each centroid corresponds to the representative color of each
        cluster. The output should be a matrix with size(pixels, 1) rows and
        3 columns. The range of values should be [0, 255].
    """

    # Flatten the image to a 2D array of pixels
    pixels = pixels.reshape(-1, 3)
    
    # Randomly initialize the centroids
    centroids = pixels[np.random.choice(pixels.shape[0], K, replace=False)]
    
    for _ in range(10):
        # Assign each pixel to the nearest cluster center
        distances = np.linalg.norm(pixels[:, np.newaxis] - centroids, axis=2)
        clusters = np.argmin(distances, axis=1)
        
        # Calculate new cluster centroids
        new_centroids = np.array([pixels[clusters == k].mean(axis=0) for k in range(K)])
        
        # Check for convergence (if centroids do not change)
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids
    
    return clusters, centroids



def mykmedoids_swap(pixels, k):
    iters = 10
    pixels = pixels.reshape(-1,3)
    medoids = pixels[np.random.choice(pixels.shape[0], k, replace=False)]
    print(pixels)
    print(pixels.shape)
    samples,_ = pixels.shape

    clusters, cost = get_costs(pixels, medoids)
    count = 0

    while True:
        swap = False
        for i in range(samples):
            if not i in medoids:
                for j in range(k):
                    tmp_meds = medoids.copy()
                    tmp_meds[j] = i
                    clusters_, cost_ = get_costs(pixels, tmp_meds)

                    if cost_<cost:
                        medoids = tmp_meds
                        cost = cost_
                        swap = True
                        clusters = clusters_
                        print(f"Medoids Changed to: {medoids}.")
        count+=1

        if count>=iters:
            print("End of the iterations.")
            break
        if not swap:
            print("No changes.")
            break
    return clusters, medoids

def pam_sample(pixels, K):
    max_iter = 10
    sample_size = 100
    pixels = pixels.reshape(-1, 3)
    N, D = pixels.shape
    tol = 1e-4
    # Sample a subset of the data to run the algorithm on
    if N > sample_size:
        idx = np.random.choice(N, sample_size, replace=False)
        sampled_pixels = pixels[idx]
        print(sampled_pixels.shape)

    else:
        sampled_pixels = pixels
        sample_size = N

    # Initialize medoids randomly
    medoids_idx = np.random.choice(sample_size, K, replace=False)
    medoids = sampled_pixels[medoids_idx]
    clusters, cost = get_costs(sampled_pixels, medoids)
    prev_cost = cost
    for iteration in range(max_iter):
        print(iteration)
        # Swap medoids (only a subset)
        no_swap = True
        for i in range(sample_size):
            if i not in medoids_idx:
                for medoid_idx in range(K):
                    new_medoids = np.copy(medoids)
                    new_medoids[medoid_idx] = sampled_pixels[i]

                    clusters_,cost_ = get_costs(sampled_pixels, new_medoids)

                    
                    if cost_ < cost:
                        medoids = new_medoids
                        medoids_idx[medoid_idx] = i
                        cost = cost_
                        clusters = clusters_
                        no_swap = False

            print('here')
        if no_swap or prev_cost - cost < tol :
            print(f"Converged early at iteration {iteration}, cost: {cost}")
            break
        prev_cost = cost
    return clusters, medoids

def manhattan(p1, p2):
    return np.abs((p1[0]-p2[0])) + np.abs((p1[1]-p2[1])) + np.abs((p1[2]-p2[2]))

def distortion_function(pixels, medoids, assignments):
    """
    Calculate the total distortion (cost) for K-medoids.

    Parameters:
    pixels (ndarray): Data points, shape (N, d), where N is the number of points and d is the dimension (e.g., 3 for RGB).
    medoids (ndarray): The current medoids, shape (K, d), where K is the number of medoids.
    assignments (ndarray): Array of medoid assignments for each point, shape (N,).

    Returns:
    float: Total distortion (cost).
    """
    total_cost = 0
    N = len(pixels)
    
    # Iterate over each point
    for n in range(N):
        # Get the assigned medoid for the current point
        medoid_index = assignments[n]
        medoid = medoids[medoid_index]
        
        # Calculate the distance between the point and its medoid
        total_cost += euclidean_distance(pixels[n], medoid)
    
    return total_cost

def euclidean_distance(x1, x2):
    return np.linalg.norm(x1 - x2)

def get_costs(pixels, medoids):
    clusters = []
    cst = 0
    iter =0
    for d in pixels:
        dst = np.array([manhattan(d, md) for md in medoids])
        c = dst.argmin()
        cst+=dst.min()
        clusters.append(c)
    return clusters, cst


def mykmedoids(pixels, K):
    max_iter = 10
    sample_size = 1000
    pixels = pixels.reshape(-1, pixels.shape[-1])
    N, D = pixels.shape
    print(N)
    tol = 1e-8
    cost = float('inf')

    # Sample a subset of the data to run the algorithm on
    if N > sample_size:
        idx = np.random.choice(N, sample_size, replace=False)
        sampled_pixels = pixels[idx]
    else:
        sampled_pixels = pixels
        sample_size = N

    # Initialize medoids randomly
    medoids_idx = np.random.choice(sample_size, K, replace=False)
    medoids = sampled_pixels[medoids_idx]
    labels = np.zeros(sample_size)
    prev_cost = float('inf')

    for iteration in range(max_iter):
        # Step 1: Compute distances
        distances = compute_dist(sampled_pixels, medoids)
        labels = np.argmin(distances, axis=1)
        cost = np.sum([distances[j, labels[j]] for j in range(sample_size)])

        # Step 2: Swap medoids (only a subset)
        no_swap = True
        for medoid_idx in range(K):
            for i in range(sample_size):
                if i not in medoids_idx:
                    new_medoids = np.copy(medoids)
                    new_medoids[medoid_idx] = sampled_pixels[i]

                    new_distances = compute_dist(sampled_pixels, new_medoids)
                    new_labels = np.argmin(new_distances, axis=1)
                    new_cost = np.sum([new_distances[j, new_labels[j]] for j in range(sample_size)])

                    if new_cost < cost:
                        medoids = new_medoids
                        medoids_idx[medoid_idx] = i
                        cost = new_cost
                        no_swap = False

        if no_swap or prev_cost - cost < tol:
            print(f"Converged early at iteration {iteration}, cost: {cost}")
            break
        prev_cost = cost

    # Apply final medoids to the entire dataset
    final_distances = compute_dist(pixels, medoids)
    labels = np.argmin(final_distances, axis=1)

    return labels, medoids

    
def compute_dist(X, medoids):
    N, D = X.shape
    K = medoids.shape[0]
    distances = np.zeros((N, K))
    for i in range(N):
        for j in range(K):
            dist = 0.0
            for d in range(D):
                dist += abs(X[i, d] - medoids[j, d])
            distances[i, j] = dist
    return distances

def compute_dist_L2(X,medoids):
    N,D = X.shape
    K = medoids.shape[0]
    return np.sum((np.reshape(X,[N,1,D])-np.reshape(medoids,[1,K,D]))**2,axis=2)

def main():
    # Load the image files directly
    directory = os.getcwd()  # Get the directory of the script
    K = 2  # Number of clusters
    
    # Loop through all image files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
            image_file_name = os.path.join(directory, filename)
            print(image_file_name)
            # image_file_name = "/content/beach.bmp"  # You can replace this with the path to your image
            K = 5  # You can adjust the number of clusters here

            im = np.asarray(imageio.imread(image_file_name))
            fig, axs = plt.subplots(1, 2)

            # Apply K-medoids
            classes, centers = pam_sample(im, K)
            new_im = np.asarray(centers[classes].reshape(im.shape), im.dtype)
            imageio.imwrite(os.path.basename(os.path.splitext(image_file_name)[0]) + '_converted_mykmedoids_' + str(K) + os.path.splitext(image_file_name)[1], new_im)
            axs[0].imshow(new_im)
            axs[0].set_title('K-medoids')

            # Apply K-means
            # classes, centers = mykmeans(im, K)
            # new_im = np.asarray(centers[classes].reshape(im.shape), im.dtype)
            # imageio.imwrite(os.path.basename(os.path.splitext(image_file_name)[0]) + '_converted_mykmeans_' + str(K) + os.path.splitext(image_file_name)[1], new_im)
            # axs[1].imshow(new_im)
            # axs[1].set_title('K-means')

            plt.show()

if __name__ == '__main__':
    main()