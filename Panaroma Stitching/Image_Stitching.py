import numpy as np
import random
from scipy.ndimage import affine_transform
import cv2
from skimage.feature import plot_matches
import matplotlib.pyplot as plt

def gaussian_filter(input_image, sigma = 1):
  # padding the input image
  input_padded = np.pad(input_image, pad_width = 3*sigma, mode = 'reflect')

  # allocating memory for output image
  gaussian_filtered_image = np.zeros((input_image.shape[0], input_image.shape[1]))

  # allocating memory for gaussian filter
  gaussian_filter = np.zeros((6 * sigma + 1, 6 * sigma + 1))

  # storing the center pixel
  x_center = gaussian_filter.shape[0] // 2
  y_center = gaussian_filter.shape[1] // 2

  # adding values to the gaussian filter based on gaussian distribution
  for i in range(gaussian_filter.shape[0]):
    for j in range(gaussian_filter.shape[1]):
      x_coordinate = abs(j - y_center)
      y_coordinate = abs(i - x_center)

      term1 = 1 / (2 * np.pi * (sigma**2))
      term2 = np.exp((-x_coordinate**2 - y_coordinate**2) / (2 * sigma**2))
      gaussian_filter[j, i] = term1 * term2

  # normalizing the filter
  gaussian_filter = gaussian_filter/ np.sum(gaussian_filter)

  # performing convolution operation of gaussian filter on input image
  for i in range(input_padded.shape[0] - (gaussian_filter.shape[0] - 1)):
    for j in range(input_padded.shape[1] - (gaussian_filter.shape[1] - 1)):
      gaussian_filtered_image[i, j] = np.sum(input_padded[i: i + gaussian_filter.shape[0], j: j + gaussian_filter.shape[1]] * gaussian_filter)

  return gaussian_filtered_image

def corner_response(input_image):
  # Calculating first and second derivatives using partial derivative function
  grad_xx, grad_yy, grad_xy, grad_x, grad_y = partial_derivative(input_image)

  # specifying the window size
  window_size = 5
  threshold = 0.5
  alpha = 0.05
  corner_list = []

  for i in range(window_size, (input_image.shape[0]- window_size)):
    for j in range(window_size, (input_image.shape[1] - window_size)):
      xx = grad_xx[i-window_size: i+window_size+1, j-window_size:j+window_size+1].sum()
      yy = grad_yy[i-window_size: i+window_size+1, j-window_size:j+window_size+1].sum()
      xy = grad_xy[i-window_size: i+window_size+1, j-window_size:j+window_size+1].sum()

      term1 = (xx*yy) - (xy**2)
      term2 = (xx+yy)**2

      R = term1 - (alpha * term2)
      if R > threshold:
        corner_list.append([R,i,j])
  return corner_list

def partial_derivative(input_image):
  # defining sobel filters
  sobel_x = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
  sobel_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

  # allocating memory for gradients
  grad_x = np.zeros((input_image.shape[0], input_image.shape[1]))
  grad_y = np.zeros((input_image.shape[0], input_image.shape[1]))
  grad_xx = np.zeros((input_image.shape[0], input_image.shape[1]))
  grad_yy = np.zeros((input_image.shape[0], input_image.shape[1]))
  grad_xy = np.zeros((input_image.shape[0], input_image.shape[1]))

  # calculating derivative
  for i in range(input_image.shape[0] - (sobel_x.shape[0] - 1)):
    for j in range(input_image.shape[1] - (sobel_x.shape[1] - 1)):
      grad_x[i, j] = np.sum(input_image[i: i + sobel_x.shape[0], j: j + sobel_x.shape[1]] * sobel_x)
      grad_y[i, j] = np.sum(input_image[i: i + sobel_y.shape[0], j: j + sobel_y.shape[1]] * sobel_y)

  # calculating second derivatives for grad_xx and grad_yy
  for i in range(input_image.shape[0] - (sobel_x.shape[0] - 1)):
    for j in range(input_image.shape[1] - (sobel_y.shape[1]- 1)):
      grad_xx[i, j] = np.sum(grad_x[i: i + sobel_x.shape[0], j: j + sobel_x.shape[1]] * sobel_x)
      grad_yy[i, j] = np.sum(grad_y[i: i + sobel_y.shape[0], j: j + sobel_y.shape[1]] * sobel_y)
  # calculating second derivatives for grad_xy
  for i in range(grad_xy.shape[0] - (sobel_y.shape[0] - 1)):
    for j in range(grad_xy.shape[1] - (sobel_y.shape[1] - 1)):
      grad_xy[i, j] = np.sum(grad_x[i: i + sobel_y.shape[0], j: j + sobel_y.shape[1]] * sobel_y)

  return grad_xx, grad_yy, grad_xy, grad_x, grad_y

def corners(input_image):
  corner_list = corner_response(input_image)
  corner_list.sort(key = lambda x:x[0], reverse = True)
  sorted_clist = corner_list[:1000]

  corner_image = np.zeros((input_image.shape[0], input_image.shape[1]))
  for r,x,y in sorted_clist:
    corner_image[x,y] = input_image[x,y]
  return corner_image

def NonMaximumSuppression(inputImage):
  corner_image = corners(inputImage)
  inputImage = corner_image
  inputImagepadded = np.pad(inputImage, pad_width = 1, mode = 'constant', constant_values = [0, 0])

  nmsCoords = []
  for i in range(inputImagepadded.shape[0] - 3):
    for j in range(inputImagepadded.shape[1] - 3):
      currentPatch = inputImagepadded[i: i + 3, j: j + 3].flatten()
      if currentPatch[4] != 0:
        if currentPatch[4] == max(currentPatch):
          nmsCoords.append([i + 1, j + 1])

  inputImageNMS = np.zeros((inputImage.shape[0], inputImage.shape[1]))
  for coordinate in nmsCoords:
    inputImageNMS[coordinate[0], coordinate[1]] = inputImage[coordinate[0], coordinate[1]]

  return inputImageNMS, corner_image

# Keypoint similarity
def SSD(final_img1, img1, final_img2, img2):
  similarities = []
  for i in range(len(final_img1)):
    x1 = final_img1[i][0]
    y1 = final_img1[i][1]
    for j in range(len(final_img2)):
      x2 = final_img2[j][0]
      y2 = final_img2[j][1]
      s = abs(img1[x1,y1] - img2[x2,y2])
      similarities.append([s,x1,y1,x2,y2])
  similarities.sort(key=lambda x: x[0])

  temp = []
  for i in range(len(similarities)):
    if similarities[i][0]==0:
      temp.append(similarities[i])
  random_choices = np.random.choice(len(temp), 20)
  sim_list = []
  for choice in random_choices:
    sim_list.append(similarities[choice])
  return np.array(similarities)

# function to implement RANSAC
def ransacAffine(img1, keypoints_img1, img2, keypoints_img2, Points = 20):
  data = SSD(keypoints_img1, img1, keypoints_img2, img2)
  #data = np.delete(sim_list_SSD, 0, axis = 1).astype('int')

  match Points:
    case 20:
      data = sorted(data, key=lambda x: x[0])
      data = np.delete(data, 0, axis = 1).astype('int')
      putative = data[:20]
      putative = np.array(putative)
    case 30:
      data = np.delete(data, 0, axis = 1).astype('int')
      putative = random.sample(data.tolist(), 30)
      putative = np.array(putative)
    case 50:
      data = sorted(data, key=lambda x: x[0])
      data = np.delete(data, 0, axis = 1).astype('int')

      putative20 = data[:20]
      putative = np.array(putative20)

      putative30 = random.sample(data.tolist(), 30)
      putative30 = np.array(putative30)

      putative = np.vstack((putative20, putative30))

  putativeL = putative[:, 0:2]
  putativeR = putative[:, 2:4]

  bestInlierCount = 0
  bestInliers = []
  bestM = []
  bestT = []
  ctr = 1

  while True:
    print(f"Epoch: {ctr}")
    # taking random samples
    randomPointIdx = random.sample(range(len(putative)), 3)
    randomPointsImg1 = []
    randomPointsImg2 = []

    for idx in randomPointIdx:
      randomPointsImg1.append(putativeL[idx])
      randomPointsImg2.append(putativeR[idx])
    #randomPointsImg1 = np.array(randomPointsImg1)
    #randomPointsImg2 = np.array(randomPointsImg2)
    
    # obtaining M and t
    aMatrix = np.array(
          [[randomPointsImg1[0][0], randomPointsImg1[0][1], 0 , 0, 1, 0],
          [0, 0, randomPointsImg1[0][0], randomPointsImg1[0][1], 0, 1],
          [randomPointsImg1[1][0], randomPointsImg1[1][1], 0 , 0, 1, 0],
          [0, 0, randomPointsImg1[1][0], randomPointsImg1[1][1], 0, 1],
          [randomPointsImg1[2][0], randomPointsImg1[2][1], 0 , 0, 1, 0],
          [0, 0, randomPointsImg1[2][0], randomPointsImg1[2][1], 0, 1]],
          )

    bMatrix = np.array(
            [randomPointsImg2[0][0],
            randomPointsImg2[0][1],
            randomPointsImg2[1][0],
            randomPointsImg2[1][1],
            randomPointsImg2[2][0],
            randomPointsImg2[2][1]],
            )
    
    sol = np.linalg.lstsq(aMatrix, bMatrix, rcond = -1)[0]
    M = sol[0:4].reshape(2, 2)
    t = sol[4:6]
    
    # determining reprojected points using M and t
    reprojectedPoints = []
    for i in range(putative.shape[0]):
      x, y = np.dot(M, putative[i][0:2]) + t
      reprojectedPoints.append([int(x), int(y)])
    reprojectedPoints = np.array(reprojectedPoints)

    currentInlierCount = 0
    currentInliers = []
    currentInliersIdx = []
    
    # checking inliers
    for i in range(reprojectedPoints.shape[0]):
      thresh = np.sqrt((putative[i][0] - putative[i][2])**2 + (putative[i][1] - putative[i][3])**2)**2
      distance = np.sqrt((putative[i][0] - reprojectedPoints[i][0])**2 + (putative[i][1] - reprojectedPoints[i][1])**2)**2
      if distance < thresh:
        currentInlierCount += 1
        currentInliersIdx.append(i)
        currentInliers.append([putative[i][0], putative[i][1]])

    # updating best inliers
    print("CurrentInlierCount: ",currentInlierCount)
    if currentInlierCount >= bestInlierCount:
      bestInlierCount = currentInlierCount
      bestInliers = currentInliers
      bestInliersIdx = currentInliersIdx
      bestM = M
      bestT = t

    # exit condition
    if currentInlierCount >= int(0.7 * len(putative)):
      break

    ctr += 1

  finalInliers = []
  for idx in bestInliersIdx:
    finalInliers.append(putative[idx].tolist())
  finalInliers = np.array(finalInliers)

  # determining number of iterations
  p = bestInlierCount / len(putative)
  e = 1 - p
  s = 3
  N = int(np.round(np.log(1-p) / np.log(1 - (1-e)**s)))

  # fitting model again with inliers
  print("\nFitting model with only Inliers")
  finalM, finalT = fit(finalInliers, N)    
    
  return finalM, finalT, bestInliersIdx

# function to fit model on inliers only
# here N is the number of iterations
def fit(finalInliers, N):
  iters = 1
  while  iters <= 3:
    print(f"Epoch: {iters}")
    randomPointIdx = random.sample(range(len(finalInliers)), 3)
    randomPointsImg1 = []
    randomPointsImg2 = []

    for idx in randomPointIdx:
      randomPointsImg1.append(finalInliers[idx][0:2])
      randomPointsImg2.append(finalInliers[idx][2:4])
    randomPointsImg1 = np.array(randomPointsImg1)
    randomPointsImg2 = np.array(randomPointsImg2)

    aMatrix = np.array(
          [[randomPointsImg1[0][0], randomPointsImg1[0][1], 0 , 0, 1, 0],
          [0, 0, randomPointsImg1[0][0], randomPointsImg1[0][1], 0, 1],
          [randomPointsImg1[1][0], randomPointsImg1[1][1], 0 , 0, 1, 0],
          [0, 0, randomPointsImg1[1][0], randomPointsImg1[1][1], 0, 1],
          [randomPointsImg1[2][0], randomPointsImg1[2][1], 0 , 0, 1, 0],
          [0, 0, randomPointsImg1[2][0], randomPointsImg1[2][1], 0, 1]],
          )

    bMatrix = np.array(
            [randomPointsImg2[0][0],
            randomPointsImg2[0][1],
            randomPointsImg2[1][0],
            randomPointsImg2[1][1],
            randomPointsImg2[2][0],
            randomPointsImg2[2][1]],
            )
  
    sol = np.linalg.lstsq(aMatrix, bMatrix, rcond = -1)[0]
    finalM = sol[0:4].reshape(2, 2)
    finalT = sol[4:6]

    iters += 1

  return finalM, finalT

# main function
def main():
  print("\nLoading Images")
  image1 = cv2.imread("uttower_left.jpg")
  image2 = cv2.imread("uttower_right.jpg")
  img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
  img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

  # conv2D gaussian filter
  print("\nPerforming Gaussian Filtering")
  sigma = 1
  gaussian_filtered_image1 = gaussian_filter(img1, sigma  = sigma)
  gaussian_filtered_image2 = gaussian_filter(img2, sigma = sigma)

  # Extracing the corner locations
  print("\nFinding Corners and Applying Non Maximum Suppression")
  final_image1, corner_image1 = NonMaximumSuppression(gaussian_filtered_image1)
  final_image2, corner_image2 = NonMaximumSuppression(gaussian_filtered_image2)

  # Extracting keypoints
  print("\nExtracting Keypoints")
  keypoints_img1 = np.argwhere(final_image1>0)
  keypoints_img2 = np.argwhere(final_image2>0)

  sim_list_SSD = SSD(keypoints_img1, img1, keypoints_img2, img2)
  sim_list_SSD = np.delete(sim_list_SSD, 0, axis = 1).astype('int')

  # change Points parameter to either 20(best 20 based on similarity) 
  # or 30 (random 30) or 50(aggregate) to test with different putative correspondences
  print("\nPerforming RANSAC")
  M, t, idxs = ransacAffine(img1, keypoints_img1, img2, keypoints_img2, Points = 50)
  transformedInput = affine_transform(img1, M, t)
  cv2.imwrite('Stitched.png', transformedInput + img2)

  print("\nPlotting Inlier Matches")
  img1Inliers = []
  img2Inliers = []
  for id in idxs:
    img1Inliers.append(sim_list_SSD[id][0:2].tolist())
    img2Inliers.append(sim_list_SSD[id][2:4].tolist())
  img1Inliers = np.array(img1Inliers)
  img2Inliers = np.array(img2Inliers)

  connect = list(range(len(img1Inliers)))
  fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(30,30))
  plt.gray()
  plt.title("Inlier Matches (Aggregate putative correspondences)")
  plot_matches(ax, img1, img2, img1Inliers, img2Inliers, np.column_stack((connect, connect)), matches_color='g')
  
main()

## NOTE : We tried performing RANSAC using our function as well as the built-in function from skimage library and the output is the same for both the cases.
## The expected number of epoch train RANSAC on the keypoints is around 27 and the observed number of epochs is 4.After running the code multiple times, since it takes random 3 points from the putative correspondences, it might result in an exact output similar to the skimage output. Since the number of keypoints selected were top 1000 in 1st problem, we don’t have enough correspondences to map the common parts between both the images. Since the tree is the common part between both the images, those keypoints were eliminated in problem one. This results in extreme warping as the model takes correspondences from the non similar sections of the image.
