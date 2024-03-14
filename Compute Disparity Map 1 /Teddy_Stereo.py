"""
Authors:
    - Atharv Subhekar (CWID: 20015840)
    - Prasad Naik (CWID: 20016345)
"""
# ========== importing libraries ==========
import cv2
import numpy as np

# ========== rank transform function ==========
def rankTransform(inputImage, windowSize = 5):
  """
    input: 
        - input image
        - window size
    
    function: 
        - computes rank transform on the input image
    
    returns:
        - rank transformed image
  """

  inputImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
  rankTransform = np.zeros(inputImage.shape)

  # looping through image to compute rank transform
  for i in range(inputImage.shape[0] - windowSize):
    for j in range(inputImage.shape[1] - windowSize):
        # extracting windows from input image
      currentWindow = inputImage[i: i + windowSize, j: j + windowSize]
      currentWindowRavel = currentWindow.ravel()
      # computing rank of center element of the window
      rank = np.where(currentWindowRavel < currentWindowRavel[(windowSize**2) // 2])[0].shape[0]
      rankTransform[i + 2, j + 2] = rank

  return rankTransform

# ========== disparity map function ==========
def computeDisparityMap(leftImage, rightImage, windowSize = 15):
  """
  input:
      - left teddy image
      - right teddy image
      - window size
  
  function:
      - computes rank transform on the input images using the rankTransform function
      - computes disparity map using the rank transformed images
      - disparity range is predefined from 0 to 63
      
  returns:
      - disparity map
  """
  
  # computing rank transform of images
  print("\nComputing Rank Transforms of Left and Right Images")
  leftImageRankTransform = rankTransform(leftImage, 5)
  rightImageRankTransform = rankTransform(rightImage, 5)
  print("Computing Rank Transforms of Left and Right Images...Done")

  # allocating memory for disparity map
  disparityMap = np.zeros(leftImageRankTransform.shape)
  
  print("\nComputing Disparity Map...This may take a while")
  # looping through the images with the given window size to compute disparity map
  for i in range(disparityMap.shape[0]):
    for j in range(disparityMap.shape[1]):
      # dispVal stores the disparity value. maxDisparity stores the highest disparity recorded yet
      dispVal = -1
      maxDisparity = 99999
      for d in range(64):
        # creating windows
        currentWindowLeft = leftImageRankTransform[i: i + windowSize, j + d: j + d + windowSize]
        currentWindowRight = rightImageRankTransform[i: i + windowSize, j: j + windowSize]

        # taking sum of absolute differences in left and right image windows
        if currentWindowLeft.shape[1] == currentWindowRight.shape[1]:
          absoluteVal = abs(currentWindowLeft - currentWindowRight).sum()
        else:
          x = np.zeros((currentWindowRight.shape[0], abs(currentWindowLeft.shape[1] - currentWindowRight.shape[1])))
          currentWindowLeft = np.append(currentWindowLeft, x, axis = 1)
          absoluteVal = abs(currentWindowLeft - currentWindowRight).sum()

        # updating disparity parameters and assigning values to disparity map
        if absoluteVal < maxDisparity:
          dispVal = d
          maxDisparity = absoluteVal
      disparityMap[i, j] = dispVal
  
  print("Computing Disparity Map...Done")
  return disparityMap

# ========== error rate function ==========
def errorRate(groundTruth, predictedMap):
  """
  input:
      - ground truth disparity map
      - predicted disparity map
      
  function:
      - loops through the two images
      - divides the ground truth value at each pixel by 4 and compares
      to the corresponding value in the predicted disparity map
      - if the difference is greater than 1, then records an error
      - computes error percentage
      
  returns:
      - error percentage
  """
  groundTruth = cv2.cvtColor(groundTruth, cv2.COLOR_BGR2GRAY)
  errorPixelCount = 0
  
  # calculating error rate between ground truth and predicted disparity map
  for i in range(groundTruth.shape[0]):
    for j in range(groundTruth.shape[1]):
      if abs(round(groundTruth[i, j] / 4) - predictedMap[i, j]) > 1:
        errorPixelCount += 1

  # calculating error percentage
  error = (errorPixelCount / (groundTruth.shape[0] * groundTruth.shape[1])) * 100
  return error

# ========== main function ==========
def main():
    # reading images
    teddyLeft = cv2.imread('teddy/teddyL.pgm')
    teddyRight = cv2.imread('teddy/teddyR.pgm')
    groundTruth = cv2.imread('teddy/disp2.pgm')
    
    # computing disparity maps and error rate (3 x 3)
    print("\n===== Computing Disparity Map and Error Rate (3 x 3 Window)...Please Wait ======")
    
    mapDisparity3 = computeDisparityMap(teddyLeft, teddyRight, 3)
    
    print("\nSaving Disparity (3 x 3) Map | Filename: win3Disparity.pgm")
    cv2.imwrite('win3Disparity.pgm', mapDisparity3)
    print("File Saving...Done")
    
    print("\nComputing Error (3 x 3 Disparity Map)")
    error3 = errorRate(groundTruth, mapDisparity3)
    print(f"Error Rate (3 x 3 Disparity Map): {error3}")
    
    print("\n===== Computing Disparity Map and Error Rate (3 x 3 Window)...Done ======")
    
    # computing disparity maps and error rate (15 x 15)
    print("\n===== Computing Disparity Map and Error Rate (15 x 15 Window)...Please Wait =====")
    mapDisparity15 = computeDisparityMap(teddyLeft, teddyRight, 15)
    
    print("\nSaving Disparity (15 x 15) Map | Filename: win15Disparity.pgm")
    cv2.imwrite('win15Disparity.pgm', mapDisparity15)
    print("File Saving...Done")
    
    print("\nComputing Error (15 x 15 Disparity Map)")
    error15 = errorRate(groundTruth, mapDisparity15)
    print(f"Error Rate (15 x 15 Disparity Map): {error15}")
    
    print("\n===== Computing Disparity Map and Error Rate (15 x 15 Window)...Done =====")
    
if __name__ == "__main__":
    main()
    
    