import cv2
import numpy as np
import matplotlib.pyplot as plt
#read the image
img = cv2.imread(r"C:\Users\dell\Desktop\W10_007.tif")                                 
#my real image
print(img)
#cv2.imshow("img",img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# threshold input image using otsu thresholding as mask and refine with morphology
ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) 
kernel = np.ones((9,9), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# put thresh into 
result = img.copy()
result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
result[:, :, 3] = mask

# save resulting masked image

cv2.imshow('retina_masked.png', result)




'''

thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]

# Find contour and sort by contour area
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

# Find bounding box and extract ROI
x,y,w,h = cv2.boundingRect(cnts[0])
ROI = img[y:y+h, x:x+w]

cv2.imshow('ROI',img)
#cv2.imwrite('ROI.png',ROI)

'''

grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


################first way
'''
# create a binary thresholded image
#cv2.threshold(src, thresh, maxval, type[, dst])
_, binary = cv2.threshold(grayscale, 80, 200, cv2.THRESH_BINARY_INV)
# find the contours from the thresholded image
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# draw all contours
mask = np.zeros_like(img)


image = cv2.drawContours(mask, contours, -1, (0, 255, 0), 3)
out = np.zeros_like(img) # Extract out the object and place into output image
out[mask == 255] = img[mask == 255]


# Now crop
(y, x) = np.where(mask == 255)
(topy, topx) = (np.min(y), np.min(x))
(bottomy, bottomx) = (np.max(y), np.max(x))
out = out[topy:bottomy+1, topx:bottomx+1]

# Show the output image

cv2.imshow('Output', out)
cv2.waitKey(0)
cv2.destroyAllWindows()
plt.imshow(image)
plt.show()

'''
#####second way
'''
# perform edge detection
edges = cv2.Canny(grayscale, 30, 100)
# detect lines in the image using hough lines technique
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 60, np.array([]), 50, 5)
# iterate over the output lines and draw them
for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(img, (x1, y1), (x2, y2), (20, 220, 20), 3)
# show the image
plt.imshow(img)
plt.show()
'''







#calculating histogram
#cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])
histg = cv2.calcHist([result],[2],None,[256],[0,256])

'''
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
    plt.show()
'''


#cv2.imshow("img",img)
#ploting histogram
plt.plot(histg)
plt.show()