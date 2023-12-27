'''
CS558 558-A Computer Vision Homework 4 Problem 1

Team Members:
    Prasad Naik CWID: 20016345
    Atharv Subhekar CWID: 20015840

'''
#%%==========Importing Libraries==========
import cv2
import numpy as np
import random
import os

#%%==========Defining Functions==========

# function to generate random points
def generateRandomPoints(inputImage, k = 10):
  pixelData = []

  H, W, C = inputImage.shape

  randomX10 = random.sample(range(H), k)
  randomY10 = random.sample(range(W), k)

  randomK10 = list(zip(randomX10, randomY10))

  for point in randomK10:
    pixelData.append([point[0], point[1], inputImage[point[0], point[1], 0], inputImage[point[0], point[1], 1], inputImage[point[0], point[1], 2]])

  return np.array(pixelData)

# function to assign clusters based on distance to each centroid
def assignClusters(inputData, centroids):
  distance2Centroids = []
  for cent in centroids:
    distance = np.sqrt((inputData[2] - cent[2])**2 + (inputData[3] - cent[3])**2 + (inputData[3] - cent[3])**2)
    distance2Centroids.append(distance)

  return distance2Centroids.index(min(distance2Centroids))

def KMeans(inputImage, k = 10, maxIterations = 100):
    # initiating random centroids
    print("Generating Random Centroids...")
    initialCentroids = generateRandomPoints(inputImage, 10)
    print("Generating Random Centroids...Done")
    
    # while loop to find optimal centroids
    print("\nFinding Optimal Centroids...")
    counter = 1
    while maxIterations > 0: # while loop starts here
      print(f'\n====================Iteration: {counter}====================')
      print(f"Current Centroids (Cluster 1 - 10)(Format: [x, y, r, g, b]): \n{initialCentroids}\n")
      # dictionary to store points belonging to each centroid
      clusterStorage = {i: [] for i in range(k)}
      # loop to assign cluster to each pixel in the image based on rgb values
      for i in range(inputImage.shape[0]): # loop starts here
        for j in range(inputImage.shape[1]): # loop starts here
          # we store x, y, r, g, b values in currentPixelData but only use the RGB values in the assignClusters function
          currentPixelData = [i, j, inputImage[i, j, 0], inputImage[i, j, 1], inputImage[i, j, 2]]
          # assignCluster function returns the cluster number to which the current pixel belongs to
          cluster = assignClusters(currentPixelData, initialCentroids)
          # storing the pixel data in that cluster key in cluster storage
          clusterStorage[cluster].append(currentPixelData)
          # loop ends here
      # loop ends here
      
      # computing new centroids based on the current set of pixels in a cluster
      newCentroids = [0] * 10
      # looping through cluster storage to calculate new centroids based on mean
      for i in range(k): # loop starts here
        currentClusterCoordinates = clusterStorage[i]
        # if there's only one pixel belonging to a cluster, then that pixel becomes the new centroid
        # else take the mean of all the pixels
        if len(currentClusterCoordinates) > 0:
          if len(currentClusterCoordinates) > 1:
            newCentroids[i] = np.mean(currentClusterCoordinates, axis = 0, dtype = 'int')
          else:
            newCentroids[i] = currentClusterCoordinates[0]
        else:
          newCentroids[i] = initialCentroids[i]
      newCentroids = np.array(newCentroids)
      # loop ends here
      
      # checking if new centroids are same as the previous centroids
      if (newCentroids == initialCentroids).all() == False:
        initialCentroids = newCentroids
        counter += 1
        maxIterations -= 1
        print("Optimal Centroids Not Found. Reiterating...")
      else:
        print("Optimal Centroids Found. KMeans Converged\n")
        break
    # while loop ends here
    
    print("\nComputing RGB Values of the final image")
    # taking average of rgb values of all the pixels belonging to a cluster
    inputImageCopy = np.copy(inputImage)
    # looping through each cluster in cluster storage and storing rgb values
    for i in range(k): # loop starts here
        r = []
        g = []
        b = []
        
        currentClusterCoordinates = clusterStorage[i]
        for coordinate in currentClusterCoordinates: # loop starts here
            r.append(inputImage[coordinate[0], coordinate[1], 0])
            g.append(inputImage[coordinate[0], coordinate[1], 1])
            b.append(inputImage[coordinate[0], coordinate[1], 2])
        # loop ends here
        
        # taking the average of rbg values
        rVal = np.mean(r)
        gVal = np.mean(g)
        bVal = np.mean(b)
        
        # replacing the current rgb values with average of rgb values in a cluster
        for coordinate in currentClusterCoordinates: # loop starts here
            inputImageCopy[coordinate[0], coordinate[1], 0] = rVal
            inputImageCopy[coordinate[0], coordinate[1], 1] = gVal
            inputImageCopy[coordinate[0], coordinate[1], 2] = bVal
        # loop ends here
    # loop ends here
    print("Computing RGB values of the final image...Done")
    
    # saving the output image
    if os.path.exists('hw4P1Output.png'):
        print("File Already Exists. Overwriting New Results")
        os.remove('hw4P1Output.png')
        cv2.imwrite('hw4P1Output.png', inputImageCopy)
    else:
        print("Saving Segmented Image by the name: 'hw4P1Output.png'")
        cv2.imwrite('hw4P1Output.png', inputImageCopy)

#%%==========Main Function==========
def main():

    # loading input image
    white_tower = cv2.imread('white-tower.png')
    
    # calling the KMeans Function
    KMeans(white_tower, k = 10, maxIterations = 100)

main()
    
    