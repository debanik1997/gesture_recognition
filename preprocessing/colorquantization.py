import numpy as np
import cv2
from matplotlib import pyplot as plt
from tools import remove_background
from skimage.measure import label   

def getLargestCC(img):
    print(img)
    new_img = np.zeros_like(img[0])                                        # step 1
    for val in np.unique(img[0])[1:]:                                      # step 2
        mask = np.uint8(img == val)                                     # step 3
        labels, stats = cv2.connectedComponentsWithStats(mask, 4)[1:3]  # step 4
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])      # step 5
        new_img[labels == largest_label] = val
    return new_img

cap_region_x_begin = 0.5  # start point/total width
cap_region_y_end = 0.8  # start point/total width
learningRate = 0
bgModel = cv2.createBackgroundSubtractorMOG2(0, 50)
threshold = 60
blurValue = 41  # GaussianBlur parameter

img = cv2.imread('../data_creation/data/1_Hello/wave_1.png')
frame = cv2.bilateralFilter(img, 5, 50, 100)

cv2.imshow('frame',frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

frame = frame.reshape((-1,3))

# convert to np.float32
frame = np.float32(frame)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 4.0)
K = 3

ret,label,center = cv2.kmeans(frame,K,None,criteria,20, cv2.KMEANS_PP_CENTERS)

center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))

cv2.imwrite('before_isolation.png', res2)
cv2.imshow('before_isolation',res2)
cv2.waitKey(0)
cv2.destroyAllWindows()

for i in range(len(label)):
    if label[i][0] == 1:
        continue
    else:
        label[i][0] = 0
        
# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))


cv2.imwrite('res2.png', res2)
cv2.imshow('res2',res2)
cv2.waitKey(0)
cv2.destroyAllWindows()

img2 = remove_background(res2, bgModel, learningRate) #################

# convert the image into binary image
gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)

ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

cv2.imwrite('thresh_2.png', thresh)
cv2.imshow('thresh', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()


