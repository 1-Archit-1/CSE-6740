import numpy as np
import imageio
import os
import matplotlib.pyplot as plt

def euclidean_distance(x1, x2):
    return np.linalg.norm(x1 - x2, ord=2)

def manhattan_distance(x1, x2):
    return np.linalg.norm(x1 - x2, ord=1)

def distortion_function(pixels, medoids, assignments):
    """
    Calculate the total distortion (cost) for K-medoids.
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

def mykmedoids(pixels, K, max_iter=100):
    """
    K-medoids algorithm using Partitioning Around Medoids (PAM)
    
    Input:
        pixels: data set, where each row is a data point (e.g., for images it represents RGB).
        K: the number of desired clusters.
        max_iter: maximum number of iterations.
        
    Output:
        clusters: array of cluster assignments for each data point.
        medoids: array of K medoids (representative points for each cluster).
    """
    N = pixels.shape[0]
    d = pixels.shape[1]
    pixels = pixels.reshape(-1, 3)
    # Randomly initialize medoids
    medoid_indices = np.random.choice(N, K, replace=False)
    medoids = pixels[medoid_indices]
    
    for _ in range(max_iter):
        # Step 1: Assign each point to the closest medoid
        distances = np.array([[euclidean_distance(p, m) for m in medoids] for p in pixels])
        assignments = np.argmin(distances, axis=1)
        
        # Step 2: Update medoids
        new_medoids = []
        for k in range(K):
            cluster_points = pixels[assignments == k]
            if len(cluster_points) == 0:
                continue
            # Find the new medoid by minimizing the cost
            min_cost = np.inf
            best_medoid = None
            for candidate in cluster_points:
                cost = np.sum([euclidean_distance(candidate, point) for point in cluster_points])
                if cost < min_cost:
                    min_cost = cost
                    best_medoid = candidate
            new_medoids.append(best_medoid)
        
        new_medoids = np.array(new_medoids)
        
        # Check for convergence
        if np.array_equal(medoids, new_medoids):
            break
        
        medoids = new_medoids

    # Final assignment
    distances = np.array([[euclidean_distance(p, m) for m in medoids] for p in pixels])
    assignments = np.argmin(distances, axis=1)
    
    return assignments, medoids

def main():
    # Load the image files directly

    directory = "/home/archit/Desktop/CDA-Assignments/hw1/hw1"  # Get the directory of the script
    K = 5  # Number of clusters
    # Loop through all image files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(('.png', 'g.jpg', 'r.jpeg', 'as.bmp', '.tiff', '.gif')):
            image_file_name = os.path.join(directory, filename)
            print(image_file_name)
            # image_file_name = "/content/beach.bmp"  # You can replace this with the path to your image
            K = 5  # You can adjust the number of clusters here

            im = np.asarray(imageio.imread(image_file_name))
            fig, axs = plt.subplots(1, 2)

            # Apply K-medoids
            classes, centers = mykmedoids(im, K)
            new_im = np.asarray(centers[classes].reshape(im.shape), im.dtype)
            imageio.imwrite(os.path.basename(os.path.splitext(image_file_name)[0]) + '_converted_mykmedoids_' + str(K) + os.path.splitext(image_file_name)[1], new_im)
            axs[0].imshow(new_im)
            axs[0].set_title('K-medoids')

            # # Apply K-means
            # classes, centers = mykmeans(im, K)
            # new_im = np.asarray(centers[classes].reshape(im.shape), im.dtype)
            # imageio.imwrite(os.path.basename(os.path.splitext(image_file_name)[0]) + '_converted_mykmeans_' + str(K) + os.path.splitext(image_file_name)[1], new_im)
            # axs[1].imshow(new_im)
            # axs[1].set_title('K-means')

            plt.show()

if __name__ == '__main__':
    main()
