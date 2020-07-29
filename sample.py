################################IMPORTS################################################

import cv2
import numpy as np
import matplotlib.pyplot as plt


###################################FUNCTIONS############################################


def find_peaks(a):
    x = np.array(a)
    maxn = np.max(a)
    lenght = len(a)
    ret = []
    for i in range(lenght):
        ispeak = True
        if i-1 > 0:
            ispeak &= (x[i] > 1 * x[i-1])
        if i+1 < lenght:
            ispeak &= (x[i] > 1 * x[i+1])

        ispeak &= (x[i] > 0.05 * maxn)
        if ispeak:
            ret.append(i)
    return ret


########################MAIN#######################################################

#read the image
img = cv2.imread(r"C:\Users\dell\Desktop\project\project\Dr._Viswanath\3_us_6144x4415pixels_113dpi_24bitdepth\W10_017_3ms_(7).tif")                                 
#my real image
#print(img)
#cv2.waitKey(0)
histg = cv2.calcHist([img],[2],None,[256],[0,256])
a=[]
#for i in range(len(histg)):
for i in range(90-21):
    a.append(histg[i+21][0])



#Croped Image

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
kernal = np.ones((5,5),np.uint8)
opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernal)
'''
x=0
for i in range(300):
    if(opening[-i-1][300] >= 40):
        x=i
        break
img = opening[0:-x,:] 
'''
img=opening[0:4100,:]


'''
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]

# Find contour and sort by contour area
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

# Find bounding box and extract ROI
x,y,w,h = cv2.boundingRect(cnts[1])
ROI = img[y:y+h, x:x+w]

img = blur = cv2.blur(img,(5,5))


'''
#resize the image
 
 
scale_percent = 20 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

# resize image
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
 



# save resulting masked image

cv2.imshow('W10_004.png', resized)

resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

histg = cv2.calcHist([resized],[2],None,[256],[0,256])

plt.plot(histg)
plt.show()


'''
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()

'''

#print(histg)
a=[]
#for i in range(len(histg)):
for i in range(70):
    a.append(histg[i][0])
q=find_peaks(a)
#print(resized)
print(q)
t =2
k=0.25
lower_black = np.array([q[t]-k,q[t]-k,q[t]-k], dtype = "uint16")
upper_black = np.array([q[t]+k,q[t]+k,q[t]+k], dtype = "uint16")
black_mask = cv2.inRange(resized, lower_black, upper_black)
# Change image to red where we found range
#resized[black_mask>0]=(0,0,255)
resized[np.where((resized==[q[t],q[t],q[t]]).all(axis=2))] = [0,0,225]
cv2.imshow("result.png",resized)


histg = cv2.calcHist([resized],[2],None,[256],[0,256])



a=[]
#for i in range(len(histg)):
for i in range(255):
    a.append(histg[i][0])

q2 = find_peaks(a)
#q2=q
plt.plot(a)
plt.show()
print(q2)
t =2
k2=0.5
k=0
lower_black = np.array([q2[t]-k2,q2[t]-k2,q2[t]-k2], dtype = "uint16")
upper_black = np.array([q2[t]+k2,q2[t]+k2,q2[t]+k2], dtype = "uint16")
black_mask = cv2.inRange(resized, lower_black, upper_black)
# Change image to red where we found range
#resized[black_mask>0]=(0,255,0)
resized[np.where((resized==[q2[t],q2[t],q2[t]]).all(axis=2))] = [0,255,0]
cv2.imshow("result.png",resized)
histg = cv2.calcHist([resized],[2],None,[256],[0,256])


a=[]
#for i in range(len(histg)):
for i in range(255):
    a.append(histg[i][0])
q2=q
q2 = find_peaks(a)
plt.plot(a)
plt.show()
print(q2)
t =3
k2=1
k=0
lower_black = np.array([q2[t]-k2,q2[t]-k2,q2[t]-k2], dtype = "uint16")
upper_black = np.array([q2[t]+k2,q2[t]+k2,q2[t]+k2], dtype = "uint16")
black_mask = cv2.inRange(resized, lower_black, upper_black)
# Change image to red where we found range
#resized[black_mask>0]=(255,0,0)
resized[np.where((resized==[q2[t],q2[t],q2[t]]).all(axis=2))] = [255,0,0]
cv2.imshow("result.png",resized)
histg = cv2.calcHist([resized],[2],None,[256],[0,256])



a=[]
#for i in range(len(histg)):
for i in range(255):
    a.append(histg[i][0])
q2=q
q2 = find_peaks(a)
plt.plot(a)
plt.show()
print(q2)
t =4
k2=0.5
k=0
lower_black = np.array([q2[t]-k2,q2[t]-k2,q2[t]-k2], dtype = "uint16")
upper_black = np.array([q2[t]+k2,q2[t]+k2,q2[t]+k2], dtype = "uint16")
black_mask = cv2.inRange(resized, lower_black, upper_black)
# Change image to red where we found range
#resized[black_mask>0]=(255,215,0)
resized[np.where((resized==[q2[t],q2[t],q2[t]]).all(axis=2))] = [255,215,0]
cv2.imshow("result.png",resized)
histg = cv2.calcHist([resized],[2],None,[256],[0,256])




plt.plot(histg)
plt.show()
