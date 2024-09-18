import os
import imageio
import numpy as np
from matplotlib import pyplot as plt
import time

def distance_metric(cluster, pixel):
    return np.linalg.norm(cluster - pixel,ord =2)/ len(cluster)
def generate_clusters(pixels, medoids,order=2):
    distances = np.linalg.norm(pixels[:, np.newaxis] - medoids, axis=2,ord=order)
    clusters = np.argmin(distances, axis=1)
    return clusters
def select_initial_medoids_unique(pixels, K):
    unique_pixels = np.unique(pixels, axis=0)  # Remove duplicate rows (pixels)
    medoids = unique_pixels[np.random.choice(unique_pixels.shape[0], K, replace=False)]
    return medoids
def select_inital_medoids_random(pixels, K):
    return pixels[np.random.choice(pixels.shape[0], K, replace=False)]

def kmeans(pixels, K):
    # Flatten the image to a 2D array of pixels
    pixels = pixels.reshape(-1, 3)
    
    # Randomly initialize the cluster centroids
    centroids = select_inital_medoids_random(pixels, K)
    iterations = 0

    while iterations<10:
        # Assign each pixel to the nearest cluster center
        clusters = generate_clusters(pixels, centroids)
        # Calculate new cluster centroids
        new_centroids= np.zeros(centroids.shape)
        for k in range(K):
            cluster = pixels[clusters == k]
            if len(cluster) == 0:
                continue
            new_centroids[k] = cluster.mean(axis=0)
        #new_centroids = np.array([pixels[clusters == k].mean(axis=0) for k in range(K)])
        # Check for convergence (if centroids do not change)
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
        iterations+=1
    
    return clusters, centroids

def kmedoids(pixels, K):
    """Function to perform K-medoids clustering on the given pixels."""
    pixels = pixels.reshape(-1, 3)  # Flatten the image to a 2D array of pixels
    #medoids = pixels[np.random.choice(pixels.shape[0], K, replace=False)]  # Randomly initialize the medoids
    medoids = select_inital_medoids_random(pixels, K) 
    clusters = generate_clusters(pixels,medoids,order=2)  # Initialize clusters
    new_medoids = np.zeros(medoids.shape)  # To track previous medoids
    iterations = 0  # Count iterations
    distance_metrics = []
    print(f'Initial {medoids}')
    while iterations < 10 and not np.array_equal(medoids, new_medoids):  # Limit iterations to 10
        total_metric = 0
        medoids =new_medoids
        for i in range(K):
            cluster = pixels[clusters == i]
            if len(cluster) == 0:
                continue
            current_min = distance_metric(cluster, medoids[i])
            sample_size = len(cluster)//20 +1
            sample_idx = np.random.choice(len(cluster), sample_size, replace=False)

            sample = cluster[sample_idx]
            for pixel in sample:
                metric = distance_metric(cluster, pixel)
                if metric < current_min:
                    current_min = metric
                    new_medoids[i] = pixel
            total_metric+= current_min
        distance_metrics.append(total_metric)
        new_clusters = generate_clusters(pixels, medoids,order=2)
        iterations+=1
        
        if min(abs(np.array(distance_metrics[-2:]) - distance_metrics[-1])) < 1e-8:
            break
        else:
            clusters = new_clusters

    for idx,medoid in enumerate(medoids):
        if idx not in clusters:
            print(f'Medoid {idx} is empty')
    return clusters, medoids


def main():
    # Load the image files directly
    directory = os.getcwd()  # Get the directory of the script
    K = 2  # Number of clusters
    
    # Loop through all image files in the directory
    time_taken = []
    for filename in os.listdir(directory):

        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
            image_file_name = os.path.join(directory, filename)
            print(image_file_name)
            K=16

            t = time.time()
            im = np.asarray(imageio.imread(image_file_name))
            fig, axs = plt.subplots(1, 2)

            #Apply K-medoids
            classes, centers = kmedoids(im, K)
            new_im = np.asarray(centers[classes].reshape(im.shape), im.dtype)
            imageio.imwrite(os.path.basename(os.path.splitext(image_file_name)[0]) + '_converted_mykmedoids_' + str(K) + os.path.splitext(image_file_name)[1], new_im)
            axs[0].imshow(new_im)
            axs[0].set_title('K-medoids')
            # print(f'classes {classes}')
            # print(f'centers {centers}')
            # time_taken.append(time.time() - t)
            # print(f'Time taken: {time.time() - t:.2f} seconds')

            #Apply K-means
            classes, centers = kmeans(im, K)
            new_im = np.asarray(centers[classes].reshape(im.shape), im.dtype)
            imageio.imwrite(os.path.basename(os.path.splitext(image_file_name)[0]) + '_converted_mykmeans_' + str(K) + os.path.splitext(image_file_name)[1], new_im)
            axs[1].imshow(new_im)
            axs[1].set_title('K-means')
            # print(f'classes {classes}')
            # print(f'centers {centers}')
            # time_taken.append(time.time() - t)
            # print(f'Time taken: {time.time() - t:.2f} seconds')
            plt.show()
    
    #Graph all the time taken for each K
    plt.plot(range(1, len(time_taken) + 1), time_taken)

if __name__ == '__main__':
    main()