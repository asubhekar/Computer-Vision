
# IMPORTING LIBRARIES
import cv2
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from sympy import *

# HELPER FUNCTIONS
input_image = cv2.imread('road.png')
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

# filtering function for Gaussian and Sobel Filtering
def Filter(input_image, sigma = 1, mode = 'Gaussian', SobelMode = 'normal'):
    # handling filter mode
    match mode:
        case 'Gaussian':
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

        case 'Sobel':
            # defining sobel filters
            sobel_x = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
            sobel_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

            # allocating memory for gradients
            grad_x = np.zeros((input_image.shape[0], input_image.shape[1]))
            grad_y = np.zeros((input_image.shape[0], input_image.shape[1]))
            grad_xx = np.zeros((input_image.shape[0], input_image.shape[1]))
            grad_yy = np.zeros((input_image.shape[0], input_image.shape[1]))
            grad_xy = np.zeros((input_image.shape[0], input_image.shape[1]))

            # allocating memory for gradient orientations
            alpha = np.zeros((input_image.shape[0], input_image.shape[1]))

            # smoothing the image before calculating gradients
            smoothed_input_image = Filter(input_image, sigma = 1, mode = 'Gaussian')

            # calculating gradients
            for i in range(input_image.shape[0] - (sobel_x.shape[0] - 1)):
              for j in range(input_image.shape[1] - (sobel_x.shape[1] - 1)):
                grad_x[i, j] = np.sum(smoothed_input_image[i: i + sobel_x.shape[0], j: j + sobel_x.shape[1]] * sobel_x)
                grad_y[i, j] = np.sum(smoothed_input_image[i: i + sobel_y.shape[0], j: j + sobel_y.shape[1]] * sobel_y)

            # calculating second derivatives
            for i in range(input_image.shape[0] - (sobel_x.shape[0] - 1)):
              for j in range(input_image.shape[1] - (sobel_y.shape[1]- 1)):
                grad_xx[i, j] = np.sum(grad_x[i: i + sobel_x.shape[0], j: j + sobel_x.shape[1]] * sobel_x)
                grad_yy[i, j] = np.sum(grad_y[i: i + sobel_y.shape[0], j: j + sobel_y.shape[1]] * sobel_y)

            # handling SobelMode
            match SobelMode:
              case 'x':
                return grad_x

              case 'y':
                return grad_y

              case 'xx':
                return grad_xx

              case 'yy':
                return grad_yy

              case 'xy':
                for i in range(grad_xy.shape[0] - (sobel_y.shape[0] - 1)):
                  for j in range(grad_xy.shape[1] - (sobel_y.shape[1] - 1)):
                    grad_xy[i, j] = np.sum(grad_x[i: i + sobel_y.shape[0], j: j + sobel_y.shape[1]] * sobel_y)

                # computing orientations
                alpha = np.zeros((input_image.shape[0], input_image.shape[1]))
                for i in range(input_image.shape[0]):
                  for j in range(input_image.shape[0]):
                    if grad_xx[i, j] != 0:
                      alpha[i, j] = math.degrees(np.arctan(grad_y[i, j] / grad_xx[i, j]))

                return grad_xy, alpha

              case 'normal':
                # computing gradient magnitude
                gradient_magnitude = np.zeros((input_image.shape[0], input_image.shape[1]))
                for i in range(gradient_magnitude.shape[0]):
                    for j in range(gradient_magnitude.shape[1]):
                        gradient_magnitude[i, j] = np.sqrt(grad_x[i, j]**2 + grad_y[i, j]**2)

                # thresholding gradient magnitude
                threshold = 79
                for i in range(gradient_magnitude.shape[0]):
                    for j in range(gradient_magnitude.shape[1]):
                        if gradient_magnitude[i, j] < threshold:
                            gradient_magnitude[i, j] = 0

                # computing gradient orientations
                for i in range(input_image.shape[0]):
                    for j in range(input_image.shape[1]):
                        if grad_x[i, j] != 0:
                            alpha[i, j] = math.degrees(np.arctan(grad_y[i, j] / grad_x[i, j]))


                return gradient_magnitude, alpha

# non maximum suppression function
def NonMaximumSuppression(input_image, sigma = 1):
    # obtaining gradient magnitude and orientations from the Filter() function
    grad_mag, orientations = Filter(input_image, sigma = 1, mode = "Sobel", SobelMode = "normal")

    # padding gradient magnitude to handle edge case during orientation check
    grad_mag_padded = np.pad(grad_mag, pad_width = 1, mode = 'reflect')

    # allocating memory for final output
    nonMaxSuppression = np.zeros((input_image.shape[0], input_image.shape[1]))

    # performing nonmaximum suppression
    for i in range(1, grad_mag.shape[0] - 1):
        for j in range(1, grad_mag.shape[1] - 1):
            current_orientation = orientations[i, j]
            if current_orientation < 0:
                current_orientation += 180

            # checking vertical, horizontal, -45, +45 orientation direction
            if 0 < current_orientation <= 22.5:
                # orientation = horizontal (check vertical neighbors)
                if grad_mag_padded[i, j] < grad_mag_padded[i - 1, j] or grad_mag_padded[i, j] < grad_mag_padded[i + 1, j]:
                    nonMaxSuppression[i, j] = 0
                else:
                    nonMaxSuppression[i, j] = grad_mag[i,j]
            elif 22.5 < current_orientation <= 67.5:
                # orientation = -45 (check +45 neighbors)
                if grad_mag_padded[i, j] < grad_mag_padded[i + 1, j - 1] or grad_mag_padded[i, j] < grad_mag_padded[i - 1, j + 1]:
                    nonMaxSuppression[i, j] = 0
                else:
                    nonMaxSuppression[i, j] = grad_mag[i, j]
            elif 67.5 < current_orientation <= 112.5:
                # orientation = vertical (check horizontal neighbors)
                if grad_mag_padded[i, j] < grad_mag_padded[i, j - 1] or grad_mag_padded[i, j] < grad_mag_padded[i, j + 1]:
                    nonMaxSuppression[i, j] = 0
                else:
                    nonMaxSuppression[i, j] = grad_mag[i, j]
            else:
                # orientation = +45 (check -45 neighbors)
                if grad_mag_padded[i, j] < grad_mag_padded[i - 1, j - 1] or grad_mag_padded[i, j] < grad_mag_padded[i + 1, j + 1]:
                    nonMaxSuppression[i, j] = 0
                else:
                  nonMaxSuppression[i, j] = grad_mag[i, j]

    return nonMaxSuppression

# PROBLEM 1: PREPROCESSING
def preprocess(input_image, sigma = 1, threshold = 55000):
  # obtaining gradients in different directions
  i_xx = Filter(input_image, sigma = 1, mode = 'Sobel', SobelMode = 'xx')
  i_yy = Filter(input_image, sigma = 1, mode = 'Sobel', SobelMode = 'yy')
  i_xy, xy_orientations = Filter(input_image, sigma = 1, mode = 'Sobel', SobelMode = 'xy')

  # calculating the determinant of Hessian
  hessian_determinant = np.zeros((input_image.shape[0], input_image.shape[1]))
  for i in range(hessian_determinant.shape[0]):
    for j in range(hessian_determinant.shape[1]):
      hessian_determinant[i, j] = (i_xx[i, j] * i_yy[i, j]) - (i_xy[i, j]**2)

  # thresholding the determinant
  hessian_threshold = threshold
  for i in range(hessian_determinant.shape[0]):
    for j in range(hessian_determinant.shape[1]):
      if hessian_determinant[i, j] < hessian_threshold:
        hessian_determinant[i, j] = 0
      else:
        hessian_determinant[i, j] = 255

  # performing NMS on hessian determinant
  hessian_NMS = NonMaximumSuppression(hessian_determinant, sigma = 1)

  return hessian_NMS

def RANSAC(input_image, keypoints, confidence = 85):
  keypoint_coordinates = []
  for i in range(keypoints.shape[0]):
    for j in range(keypoints.shape[1]):
      if keypoints[i, j] > 0:
        keypoint_coordinates.append([i, j])
  
  id1, id2 = np.random.randint(0, len(keypoint_coordinates), 2)

  # get x and y coordinates at these ids
  x1 = keypoint_coordinates[id1][0]
  y1 = keypoint_coordinates[id1][1]
  p1 = Point(x1, y1)

  x2 = keypoint_coordinates[id2][0]
  y2 = keypoint_coordinates[id2][1]
  p2 = Point(x2, y2)

  # create a line using these points
  line = Line(p1, p2)

  inlier_count = 0

  # calculate distance from the line formed by these two lines to every other keypoint
  for id in range(len(keypoint_coordinates)):
    x3 = keypoint_coordinates[id][0]
    y3 = keypoint_coordinates[id][1]
    p3 = Point(x3, y3)

    t = N(line.distance(p3))
    if t != 0 and t < 1.95:
      inlier_count += 1
    
  if inlier_count > 10:
    cv2.line(input_image, (x1, y1), (x2, y2), (255, 0, 0), 1)
    cv2.imshow("RANSAC", input_image)
    cv2.waitKey()
    cv2.destroyAllWindows()

# PROBLEM 3: HOUGH TRANSFORM
def HoughTransform(input_image, bin_dim):
  edge_detected_input_image = NonMaximumSuppression(input_image, sigma = 2)

  # defining dimensions of accumulator matrix
  max_rho = int(np.round(np.sqrt(input_image.shape[0]**2 + input_image.shape[1]** 2)))
  rho_dimensions = 2 * max_rho
  thetas = np.deg2rad(np.arange(-90, 90))
  theta_bins = (180 // bin_dim)
  accumulator_matrix = np.zeros((rho_dimensions, theta_bins))

  # filling in the accumulator matrix
  for i in range(edge_detected_input_image.shape[0]):
    for j in range(edge_detected_input_image.shape[1]):
      if edge_detected_input_image[i, j] > 0:
        counter = 0
        currBin = 0
        for k in range(len(thetas)):
          rho = i * np.cos(thetas[k]) + j * np.sin(thetas[k])
          if counter < bin_dim:
            accumulator_matrix[max_rho + int(rho), currBin] += 1
            counter += 1
          else:
            currBin += 1
            counter = 0

  # finding local maximums
  local_maximums = {}
  window = 3
  for i in range(accumulator_matrix.shape[0] - (window - 1)):
    for j in range(accumulator_matrix.shape[1] - (window - 1)):
      curr_window = accumulator_matrix[i: i + window, j: j + window]
      if np.max(curr_window) != 0:
        idx = np.unravel_index(np.argmax(curr_window), curr_window.shape)
        local_maximums[(idx[0] + i, idx[1] + j)] = curr_window[idx]

  # sorting the local maximums according to rho values to give lines with highest confidence
  local_maximums = sorted(local_maximums.items(), key = lambda item: item[1], reverse = True)
  local_maximums = local_maximums[0: 500]

  # plotting the lines
  for line in local_maximums:
    rho = line[0][0]
    theta = line[0][1]

    a = np.cos(theta)
    b = np.sin(theta)

    x0 = a * rho
    y0 = b * rho

    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))

    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    houghLines = cv2.line(input_image, (x1, y1), (x2, y2), [255, 0, 0], thickness = 1)

  return houghLines, accumulator_matrix


def main():
   # loading the input image
  input_image = cv2.imread('road.png')
  input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

  # problem 1
  keyPts = preprocess(input_image, sigma = 1, threshold = 220000)
  cv2.imshow("Keypoints", keyPts)
  cv2.waitKey()
  cv2.destroyAllWindows()

  # problem 2
  n = 0
  while n < 4:
    RANSAC(input_image, keyPts)
    n += 1

  # problem 3 
  # we perform the loading image step again since RANSAC draws lines directly on the input image
  input_image = cv2.imread('road.png')
  input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
  
  houghLines, accumulatorMatrix = HoughTransform(input_image, bin_dim = 5)
  cv2.imshow("Accumulator Matrix", accumulatorMatrix)
  cv2.imshow("Hough Lines", houghLines)
  cv2.waitKey()
  cv2.destroyAllWindows()


main()
