import concurrent.futures
import numpy as np
import pandas as pd
import time


def lists_equal(a, b):
    return np.array_equal(np.sort(a), np.sort(b))

#distance computation function with Euclidean distance
def euclidean(a, b):
    return np.sqrt(np.sum(np.square(np.subtract(a, b))))

#distance computation function with cosine similarity
def cosine(a, b):
    return 1 - np.divide(np.sum(np.multiply(a, b)),
                         np.multiply(np.sqrt(np.sum(np.square(a))),
                                     np.sqrt(np.sum(np.square(b)))))

#distance computation function with jaccard distance
def jaccard(a, b):
    return 1 - np.divide(np.sum(np.minimum(a, b)),
                         np.sum(np.maximum(a, b)))


def SSE(distance_function, X, centroids):
    result = 0
    for centroid in centroids:
        for point in X:
            result += distance_function(centroid, point)**2
    return result


def accuracy(Y, computed_Y):
    cluster_score = []
    for i in range(len(Y)):
        cluster_score.insert(i, [])
        for j in range(len(Y)):
            cluster_score[i].insert(j, 0)
    
    for i in range(len(Y)):
        cluster_score[computed_Y[i]][Y[i][0]] += 1
    
    correct = 0
    total = 0
    for i in range(len(Y)):
        winner = 0
        max_seen = 0
        for j in range(len(Y)):
            if cluster_score[i][j] > max_seen:
                winner = j
                max_seen = cluster_score[i][j]
                
        for j in range(len(Y)):
            total += cluster_score[i][j]
            if j == winner:
                correct += cluster_score[i][j]
    return correct / total


def Kmeans(distance_func, X, Y=[], K=0, centroids=np.array([]), stoppers=["Centroid unchanged"],
           maxIterations=0, task_id=""):

    if task_id == "stopper":
        ret = str(distance_func.__name__) + "\t" + str(stoppers) + "\n"
    else:
        ret = str(distance_func.__name__) + "\n"
   
    computed_Y = np.full(X.shape[0], 0)
   
    if len(stoppers) < 1 and maxIterations == 0:
        print("Missing stop criteria.")
        return
  
    if centroids.size > 0:

        if len(centroids) != K:
            print("Mismatch Found " + str(centroids) +
                  "Initial centroids with K = " + str(K))
 
    else:
        centroids = X[np.random.choice(X.shape[0], K, replace=False), :]

    start = time.time_ns()
    iterations = 0
    while True:

        old_centroids = np.copy(centroids)
        iterations += 1

        tmp_centroid_sum = np.zeros(centroids.shape)
        tmp_centroid_count = np.zeros(centroids.shape[0])
    
        for point_idx, point in enumerate(X):
            shortest_distance = float('inf')
  
            for centroid_idx, centroid in enumerate(centroids):
                distance = distance_func(point, centroid)
                if distance < shortest_distance:
                    shortest_distance = distance

                    computed_Y[point_idx] = centroid_idx

            tmp_centroid_sum[computed_Y[point_idx]] = np.add(
                tmp_centroid_sum[computed_Y[point_idx]], point)

            tmp_centroid_count[computed_Y[point_idx]] += 1

        for i in range(len(centroids)):

            if tmp_centroid_count[i] == 0:
                print("A centroid was found empty at iteration " + str(iterations))

                centroids[i] = np.copy(old_centroids[i])
            else:

                centroids[i] = np.divide(tmp_centroid_sum[i],
                                         np.full(centroids.shape[1], tmp_centroid_count[i]))

        if "Centroid unchanged" in stoppers and lists_equal(old_centroids, centroids):
            break #when there is no change in centroid position
           
        if "SSE" in stoppers and SSE(distance_func, X, centroids) \
                > SSE(distance_func, X, old_centroids):

            centroids = np.copy(old_centroids)
            break #when the SSE value increases in the next iteration

        if (maxIterations != 0 and iterations >= maxIterations) \
                or (maxIterations == 0 and iterations >= 500):
            break #when the maximum preset value (e.g., 500, here) of iteration is complete

    end = time.time_ns()

    if task_id == "task1":
        ret += "SSE = " + str(SSE(distance_func, X, centroids)) + "\n"
        ret += "Predictive accuracy = " + str(accuracy(Y, computed_Y))
    if task_id == "stopper":
        ret += str(iterations) + "\t" + str(SSE(distance_func, X, centroids)) \
            + "\t" + str(end - start) + " nsec"
    return ret


def main():

    X = pd.read_csv("./Kmeans_data/data.csv")
    Y = pd.read_csv("./Kmeans_data/label.csv")
    X = X.to_numpy(dtype=float)
    Y = Y.to_numpy(dtype=int)

  
    with concurrent.futures.ProcessPoolExecutor(max_workers=24) as executor:
        
        futures = []

        #Task 1.1 and 1.2
        futures.append(executor.submit(
            Kmeans, euclidean, X, Y=Y, K=10, task_id="task1"))
        futures.append(executor.submit(
            Kmeans, cosine, X, Y=Y, K=10, task_id="task1"))
        futures.append(executor.submit(
            Kmeans, jaccard, X, Y=Y, K=10, task_id="task1"))

        #Task 1.3
        futures.append(executor.submit(Kmeans, euclidean, X, Y=Y,
                       K=10, maxIterations=500, stoppers=["Centroid unchanged", "SSE"], task_id="stopper"))
        futures.append(executor.submit(Kmeans, cosine, X, Y=Y,
                       K=10, maxIterations=500, stoppers=["Centroid unchanged", "SSE"], task_id="stopper"))
        futures.append(executor.submit(Kmeans, jaccard, X, Y=Y,
                       K=10, maxIterations=500, stoppers=["Centroid unchanged", "SSE"], task_id="stopper"))

        #Task 1.4
        futures.append(executor.submit(Kmeans, euclidean, X, Y=Y,
                       K=10, stoppers=["Centroid unchanged"], task_id="stopper"))
        futures.append(executor.submit(Kmeans, cosine, X, Y=Y,
                       K=10, stoppers=["Centroid unchanged"], task_id="stopper"))
        futures.append(executor.submit(Kmeans, jaccard, X, Y=Y,
                       K=10, stoppers=["Centroid unchanged"], task_id="stopper"))

        futures.append(executor.submit(Kmeans, euclidean, X, Y=Y,
                       K=10, stoppers=["SSE"], task_id="stopper"))
        futures.append(executor.submit(Kmeans, cosine, X, Y=Y,
                       K=10, stoppers=["SSE"], task_id="stopper"))
        futures.append(executor.submit(Kmeans, jaccard, X, Y=Y,
                       K=10, stoppers=["SSE"], task_id="stopper"))

        futures.append(executor.submit(Kmeans, euclidean, X, Y=Y,
                       K=10, stoppers=[], maxIterations=100, task_id="stopper"))
        futures.append(executor.submit(Kmeans, cosine, X, Y=Y,
                       K=10, stoppers=[], maxIterations=100, task_id="stopper"))
        futures.append(executor.submit(Kmeans, jaccard, X, Y=Y,
                       K=10, stoppers=[], maxIterations=100, task_id="stopper"))

        
        iter_futures = iter(futures)

        print("Task 1.1 and Task 1.2")
        print(next(iter_futures).result())
        print(next(iter_futures).result())
        print(next(iter_futures).result())

        print("Task 1.3")
        print("Iterations\tSSE\tTime")
        print(next(iter_futures).result())
        print(next(iter_futures).result())
        print(next(iter_futures).result())

        print("Task 1.4")
        print("Iterations\tSSE\tTime")
        print(next(iter_futures).result())
        print(next(iter_futures).result())
        print(next(iter_futures).result())
        print(next(iter_futures).result())
        print(next(iter_futures).result())
        print(next(iter_futures).result())
        print(next(iter_futures).result())
        print(next(iter_futures).result())
        print(next(iter_futures).result())

if __name__ == "__main__":
    main()


