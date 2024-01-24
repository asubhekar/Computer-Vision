#%%==========Importing Libraries==========
import cv2
import numpy as np
import random
import os

#%%==========Defining Functions==========

# generating initial centroids
def generateSlicCentroids(inputImage, boxSize = 50):
  H, W, C = inputImage.shape
  slicCentroids = []
  # loop to find centroids at the center of each 50x50 window
  for i in range(0, H, boxSize):
    for j in range(0, W, boxSize):
      currentPatch = inputImage[i: i + 50, j: j + 50]

      fiveDSpace = []
      fiveDSpace.append(currentPatch.shape[0] // 2)
      fiveDSpace.append(currentPatch.shape[0] // 2)

      fiveDSpace.append(currentPatch[fiveDSpace[0], fiveDSpace[1], 0])
      fiveDSpace.append(currentPatch[fiveDSpace[0], fiveDSpace[1], 1])
      fiveDSpace.append(currentPatch[fiveDSpace[0], fiveDSpace[1], 2])

      fiveDSpace[0] = fiveDSpace[0] + i
      fiveDSpace[1] = fiveDSpace[1] + j

      slicCentroids.append(fiveDSpace)

  return np.array(slicCentroids)

# function to compute gradients
def computeGradients(inputImage):
  sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
  sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

  # allocating memory to save x and y gradients
  grad_x = np.zeros((inputImage.shape[0], inputImage.shape[1]))
  grad_y = np.zeros((inputImage.shape[0], inputImage.shape[1]))

  # calculating gradients
  for i in range(inputImage.shape[0] - (sobel_x.shape[0] - 1)):
    for j in range(inputImage.shape[1] - (sobel_x.shape[1] - 1)):
        grad_x[i, j] = np.sum(inputImage[i: i + sobel_x.shape[0], j: j + sobel_x.shape[1]] * sobel_x)
        grad_y[i, j] = np.sum(inputImage[i: i + sobel_y.shape[0], j: j + sobel_y.shape[1]] * sobel_y)

  # computing gradient magnitude
  gradient_magnitude = np.zeros((inputImage.shape[0], inputImage.shape[1]))
  for i in range(gradient_magnitude.shape[0]):
    for j in range(gradient_magnitude.shape[1]):
        gradient_magnitude[i, j] = np.sqrt(grad_x[i, j]**2 + grad_y[i, j]**2)

  return gradient_magnitude

# function to calculate combined gradient magnitude of RGB magnitudes
def computeGradientMagnitude(rVal, gVal, bVal):
  rGradient = computeGradients(rVal)
  gGradient = computeGradients(bVal)
  bGradient = computeGradients(gVal)

  return np.sqrt((rGradient)**2 + (gGradient)**2 + (bGradient)**2)

# function to shift the centroid in the direction of lowest gradient magnitude
def localShift(gradMag, centroids):
  for centroid in centroids:
    i = centroid[0]
    j = centroid[1]

    currentWindowGradMag = gradMag[i - 1: i + 2, j - 1: j + 2]

    x, y = np.unravel_index(np.argmin(currentWindowGradMag), currentWindowGradMag.shape)
    localShift = (x, y)

    match localShift:
      case (0, 0):
        centroid[0] = i - 1
        centroid[1] = j - 1
      case (0, 1):
        centroid[0] = i - 1
        centroid[1] = j
      case (0, 2):
        centroid[0] = i - 1
        centroid[1] = j + 1
      case (1, 0):
        centroid[0] = i
        centroid[1] = j - 1
      case (1, 1):
        centroid[0] = i
        centroid[1] = j
      case (1, 2):
        centroid[0] = i
        centroid[1] = j + 1
      case (2, 0):
        centroid[0] = i + 1
        centroid[1] = j - 1
      case (2, 1):
        centroid[0] = i + 1
        centroid[1] = j
      case (2, 2):
        centroid[0] = i + 1
        centroid[1] = j + 1

  return centroids

# function to assign centroid to a pixel based on x, y, R, G, B values.
def assignClusters(inputData, centroids):
  distance2Centroids = []
  for cent in centroids:
    distanceXY = np.sqrt((inputData[0] - cent[0])**2 + (inputData[1] - cent[1])**2)
    distanceRGB = np.sqrt((inputData[2] - cent[2])**2 + (inputData[3] - cent[3])**2 + (inputData[3] - cent[3])**2)
    distance = distanceRGB + (2 * distanceXY)
    distance2Centroids.append(distance)

  return distance2Centroids.index(min(distance2Centroids))

# function to perform SLIC segmentation
def SLIC(inputImage, maxIterations = 3):
    
    # generating centroids at the center of each 50x50 block
    print("Obtaining initial centroids")
    initialSlicCentroids = generateSlicCentroids(inputImage, 50)
    print("Obtaining initial centroids...Done")
    
    
    # locally shifting the centroid to avoid noise
    print("\nLocally shifting initial centroids")
    gradientMagnitude = computeGradientMagnitude(inputImage[:, :, 0], inputImage[:, :, 1], inputImage[:, :, 2])
    localShiftedCentroids = localShift(gradientMagnitude, initialSlicCentroids)
    print("Locally shifting initial centroids...Done")
    
    # while loop to obtain optimal local centroids
    counter = 1
    maxCount = 3
    k = localShiftedCentroids.shape[0]
    while maxIterations > 0: # while loop starts here
        print(f'\n====================Iteration: {counter}====================')
        # maintaining a dictionary to store pixels belonging to each cluster
        clusterStorage = {i: [] for i in range(localShiftedCentroids.shape[0])}
        for key in clusterStorage:
            clusterStorage[key].append(localShiftedCentroids[key])
        
        print("\nAssigning Clusters")
        # loop to assign clusters to each pixel
        for i in range(inputImage.shape[0]):
            for j in range(inputImage.shape[1]):
                currentPixelData = [i, j, inputImage[i, j, 0], inputImage[i, j, 1], inputImage[i, j, 2]]
                cluster = assignClusters(currentPixelData, localShiftedCentroids)
                clusterStorage[cluster].append(currentPixelData)
        print("Assigning Clusters...Done")
        
        print("\nComputing New Centroids")
        # computing new centroids based on the mean of pixels belonging to each cluster
        newSlicCentroids = [0] * k
        for i in range(k):
            currentClusterCoordinates = clusterStorage[i]
            if len(currentClusterCoordinates) > 0:
                if len(currentClusterCoordinates) > 1:
                    newSlicCentroids[i] = np.mean(currentClusterCoordinates, axis = 0, dtype = 'int')
                else:
                    newSlicCentroids[i] = currentClusterCoordinates[0]
            else:
                newSlicCentroids[i] = localShiftedCentroids[i]
        newSlicCentroids = np.array(newSlicCentroids)
        print("Computing New Centroids...Done")
        
        # checking if new centroids are same as previous centroids
        if (newSlicCentroids == localShiftedCentroids).all() == False:
            localShiftedCentroids = localShift(gradientMagnitude, newSlicCentroids)
            maxIterations -= 1
            if counter == maxCount:
              print("Reached Max Iterations")
            else:
              counter += 1
              print("\nOptimal Centroids Not Found. Locally Shifting Centroids and Reiterating...")
        else:
            print("\nOptimal Centroids Found. SLIC Converged")
            break
    # while loop ends here
    
    # assigning average RGB values to every pixel in the same cluster
    print("\nAssigning average RGB values to every pixel in the same cluster")
    inputImageCopy = np.copy(inputImage)
    for i in range(k):
        r = []
        g = []
        b = []
        currentClusterCoordinates = clusterStorage[i]
        for coordinate in currentClusterCoordinates:
            r.append(inputImage[coordinate[0], coordinate[1], 0])
            g.append(inputImage[coordinate[0], coordinate[1], 1])
            b.append(inputImage[coordinate[0], coordinate[1], 2])
        
        rVal = np.mean(r)
        gVal = np.mean(g)
        bVal = np.mean(b)
        
        for coordinate in currentClusterCoordinates:
            inputImageCopy[coordinate[0], coordinate[1], 0] = rVal
            inputImageCopy[coordinate[0], coordinate[1], 1] = gVal
            inputImageCopy[coordinate[0], coordinate[1], 2] = bVal
    print("Assigning average RGB values to every pixel in the same cluster...Done")
    
    # coloring border pixels white
    print("\nColoring Border Pixels Black")
    finalSLIC = np.copy(inputImageCopy)
    borderPixels = []
    for m in range(1, finalSLIC.shape[0] - 1):
        for n in range(1, finalSLIC.shape[1] - 1):
            if (finalSLIC[m, n] == finalSLIC[m - 1, n]).all() == False:
                borderPixels.append([m, n])
            elif (finalSLIC[m, n] == finalSLIC[m, n - 1]).all() == False:
                borderPixels.append([m, n])
            elif (finalSLIC[m, n] == finalSLIC[m + 1, n]).all == False:
                borderPixels.append([m, n])
            elif (finalSLIC[m, n] == finalSLIC[m, n + 1]).all() == False:
                borderPixels.append([m, n])
    
    for coordinate in borderPixels:
        finalSLIC[coordinate[0], coordinate[1], 0] = 0
        finalSLIC[coordinate[0], coordinate[1], 1] = 0
        finalSLIC[coordinate[0], coordinate[1], 2] = 0
    print("Coloring Border Pixels Black...Done")
        
    # saving the output image
    if os.path.exists('hw4P2Output.png'):
        print("File Already Exists. Overwriting New Results")
        os.remove('hw4P2Output.png')
        cv2.imwrite('hw4P2Output.png', finalSLIC)
    else:
        print("Saving Segmented Image by the name: 'hw4P2Output.png'")
        cv2.imwrite('hw4P2Output.png', finalSLIC)

#%%==========Main function==========
def main():
    # loading the input image
    wt_slic = cv2.imread('wt_slic.png')
    
    # calling the SLIC function
    SLIC(wt_slic, 3)
main()
    
    


