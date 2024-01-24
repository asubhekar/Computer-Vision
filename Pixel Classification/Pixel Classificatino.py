
#%%==========Importing Libraries==========
import cv2
import numpy as np
import random
import os

#%%==========Defining Helper Functions==========
def generateRandomCentroids(coordinates, k = 10):
  centroids = []

  random.seed(695)
  random10 = random.sample(range(len(coordinates)), k)

  for idx in random10:
    centroids.append(coordinates[idx])

  return np.array(centroids)

def computeDist(inputSet, centroids):
  distance2Clusters = []
  for cent in centroids:
    distance = np.sqrt((inputSet[2] - cent[2])**2 + (inputSet[3] - cent[3])**2 + (inputSet[4] - cent[4])**2)
    distance2Clusters.append(distance)

  return distance2Clusters.index(min(distance2Clusters))

def KMeansPixelClassification(data, k = 10):
  epoch = 1
  initialCentroids = generateRandomCentroids(data, k)

  while True:
    print("Epoch: ", epoch)
    clusterStorage = {i: [] for i in range(k)}
    for point in data:
      cluster = computeDist(point, initialCentroids)
      clusterStorage[cluster].append(point)

    newCentroids = [0] * 10
    for i in range(k):
      currentClusterCoordinates = clusterStorage[i]
      if len(currentClusterCoordinates) > 0:
        if len(currentClusterCoordinates) > 1:
          newCentroids[i] = np.mean(currentClusterCoordinates, axis = 0, dtype = 'int')
        else:
          newCentroids[i] = currentClusterCoordinates[0]
      else:
        newCentroids[i] = initialCentroids[i]
    newCentroids = np.array(newCentroids)

    if (newCentroids == initialCentroids).all() == False:
      initialCentroids = newCentroids
      epoch += 1
    else:
      print("Converged\n")
      break

  return newCentroids

def PixelClassification(test_image, skyCentroids, nonSkyCentroids):
    skyClass = 255
    nonSkyClass = 0
    testClusters = np.ones((test_image.shape[0], test_image.shape[1]), dtype = 'int')
    for i in range(test_image.shape[0]):
      for j in range(test_image.shape[1]):
        currentPixel = [i, j, test_image[i, j, 0], test_image[i, j, 1], test_image[i, j, 2]]

        distance2Sky = []
        distance2NonSky = []
        for cent in skyCentroids:
          dist = np.sqrt((currentPixel[2] - cent[2])**2 + (currentPixel[3] - cent[3])**2 + (currentPixel[4] - cent[4])**2)
          distance2Sky.append(dist)
        for cent in nonSkyCentroids:
          dist = np.sqrt((currentPixel[2] - cent[2])**2 + (currentPixel[3] - cent[3])**2 + (currentPixel[4] - cent[4])**2)
          distance2NonSky.append(dist)

        if min(distance2Sky) < min(distance2NonSky):
          testClusters[i, j] = skyClass
        else:
          testClusters[i, j] = nonSkyClass

    skyTestSegmented = np.copy(test_image)
    for i in range(skyTestSegmented.shape[0]):
      for j in range(skyTestSegmented.shape[1]):
        if testClusters[i, j] == 255:
          skyTestSegmented[i, j, 0] = 255
          skyTestSegmented[i, j, 1] = 255
          skyTestSegmented[i, j, 2] = 255
        
    return skyTestSegmented

#%%==========Main Function==========
def main():
    skyInput = cv2.imread('sky_train.jpg')
    skyMask = cv2.imread('sky_train_mask.jpg')
    
    print("\nPreparing Sky and Non-Sky Sets")
    skySet = []
    nonSkySet = []

    for i in range(skyInput.shape[0]):
      for j in range(skyInput.shape[1]):
        if skyMask[i, j, 0] == 255 and skyMask[i, j, 1] == 255 and skyMask[i, j, 2] == 255:
          skySet.append([i, j, skyInput[i, j, 0], skyInput[i, j, 1], skyInput[i, j, 2]])
        else:
          nonSkySet.append([i, j, skyInput[i, j, 0], skyInput[i, j, 1], skyInput[i, j, 2]])
    
    print("\nPerforming KMeans on Sky Set")
    skyCentroids = KMeansPixelClassification(skySet, 10)
    print("\nPerforming KMeans on NonSky Set")
    nonSkyCentroids = KMeansPixelClassification(nonSkySet, 10)
    
    #skyTest = cv2.imread('sky_test2.jpg')
    #segmentedImage = PixelClassification(skyTest, skyCentroids, nonSkyCentroids)
    
    print("\nSegmenting sky_test_1.jpg...")
    skyTest1 = cv2.imread('sky_test1.jpg')
    skyTest1_segmented = PixelClassification(skyTest1, skyCentroids, nonSkyCentroids)
    if os.path.exists('skyTest1_segmented.jpg'):
        os.remove('skyTest1_segmented.jpg')
    cv2.imwrite('skyTest1_segmented.jpg', skyTest1_segmented)
    print("Segmented image saved as 'skyTest1_segmented.jpg'")
    
    print("\nSegmenting sky_test_2.jpg...")
    skyTest2 = cv2.imread('sky_test2.jpg')
    skyTest2_segmented = PixelClassification(skyTest2, skyCentroids, nonSkyCentroids)
    if os.path.exists('skyTest2_segmented.jpg'):
        os.remove('skyTest2_segmented.jpg')
    cv2.imwrite('skyTest2_segmented.jpg', skyTest2_segmented)
    
    print("Segmented image saved as 'skyTest2_segmented.jpg'")
    print("\nSegmenting sky_test_3.jpg...")
    skyTest3 = cv2.imread('sky_test3.jpg')
    skyTest3_segmented = PixelClassification(skyTest3, skyCentroids, nonSkyCentroids)
    if os.path.exists('skyTest3_segmented.jpg'):
        os.remove('skyTest3_segmented.jpg')
    cv2.imwrite('skyTest3_segmented.jpg', skyTest3_segmented)
    print("Segmented image saved as 'skyTest3_segmented.jpg'")
    
    print("\nSegmenting sky_test_4.jpg...")
    skyTest4 = cv2.imread('sky_test4.jpg')
    skyTest4_segmented = PixelClassification(skyTest4, skyCentroids, nonSkyCentroids)
    if os.path.exists('skyTest4_segmented.jpg'):
        os.remove('skyTest4_segmented.jpg')
    cv2.imwrite('skyTest4_segmented.jpg', skyTest4_segmented)
    print("Segmented image saved as 'skyTest4_segmented.jpg'")
main()
    
