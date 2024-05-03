"""
Authors:
    - Atharv Subhekar
    - Prasad Naik
"""
#%% importing libraries
import cv2
import numpy as np
#%% computer 3D points
def compute3DPoints(inputData, depthMap, dataType = 'image'):
    S = 5000
    K = np.array([[525.0, 0, 319.5],
                  [0, 525.0, 239.5],
                  [0, 0, 1]])
    inverseK = np.linalg.inv(K)
    
    if dataType == 'image':
        points2D = []
        points3D = []
        for i in range(inputData.shape[0]):
            for j in range(inputData.shape[1]):
                if depthMap[i, j] != 0:
                    x, y, z = ((1 / S) * depthMap[i, j]) * (np.dot(inverseK, np.append(np.array([i, j]), 1)))
                    points2D.append([i, j])
                    points3D.append([x, y, z])
                    
        return np.array(points3D), np.array(points2D)
    else:
        x = inputData[0]
        y = inputData[1]
        X, Y, Z = ((1/S) * depthMap[x, y]) * (np.dot(inverseK, np.append(np.array([x, y]), 1)))
        return np.array([X, Y, Z])
    
def construcNormalMap(inputImage, depthMap):
    windowSize = 7
    windowHalfSize = windowSize // 2
    normalMap = np.zeros((inputImage.shape[0], inputImage.shape[1], inputImage.shape[2]))
    
    #points3D, points2D = compute3DPoints(inputImageGray, depthMap, dataType = 'image')
    
    for i in range(inputImage.shape[0]):
        for j in range(inputImage.shape[1]):
            if depthMap[i, j] != 0:
                pointsInWindow = []
    
                centerX, centerY = i, j
                ## this loop can be converted to a function
                for m in range(centerX - windowHalfSize, centerX + windowHalfSize + 1):
                    for n in range(centerY - windowHalfSize, centerY + windowHalfSize + 1):
                        if 0 <= m < inputImage.shape[0] and 0 <= n < inputImage.shape[1]:
                            pointsInWindow.append([m, n])
                
                pCap = []
                for point in pointsInWindow:
                    pCap.append(compute3DPoints(point, depthMap, dataType = 'point'))
                pCap = np.array(pCap)
                pCap = np.mean(pCap, axis = 0)
                
                M = np.zeros((3, 3))
                for point in pointsInWindow:
                    point3D = compute3DPoints(point, depthMap, dataType = 'point')
                    term = point3D - pCap
                    M += np.outer(term, term.T)
                M = M / (windowSize ** 2)
    
                eigenVal, eigenVec = np.linalg.eig(M)
                normal = eigenVec[:, np.argmin(eigenVal)]
                
                d = (normal[0] * points3D[i][0]) + (normal[1] * points3D[i][1]) + (normal[2] * points3D[i][2])
                if d < 0:
                    normal = -normal
                
                R, G, B = 255 * ((normal / (2 * np.linalg.norm(normal))) + [0.5, 0.5, 0.5])
                normalMap[i, j, 0] = int(R)
                normalMap[i, j, 1] = int(G)
                normalMap[i, j, 2] = int(B)
                
                print(f"Current Point: {[i, j]} | RGB Value: {int(R), int(G), int(B)}")
    
    cv2.imwrite('NormalMapping.png', normalMap.astype(int))

#%% main function here

# reading the data
print("\nReading Data...Please Wait")
inputImage = cv2.imread('rgbn.png')
inputImageGray = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
depthMap = cv2.imread('depthn.png', cv2.IMREAD_GRAYSCALE)
print("Reading Data...Complete")

# computing 3D points
print("\nComputing 3D Points...Please Wait")
points3D, points2D = compute3DPoints(inputImageGray, depthMap, dataType = 'image')
print("\nComputing 3D Points...Complete")

# performing normal mapping
print("\nPerforming Normal Mapping...Please Wait, This may take a while!")
construcNormalMap(inputImage, depthMap)
