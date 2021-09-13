import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

roi = cv.imread('pic3.jpg')
gray = cv.cvtColor(roi,cv.COLOR_BGR2GRAY)
# 1 채널 흑백이미지로 변환

binary = np.zeros((gray.shape[0],gray.shape[1]),dtype=np.uint8)

#T_max = (0,0)

# 히스토그램 생성
gray_hist = np.zeros(256)
# 정규화된 히스토그램 생성
norm_hist = np.zeros(256,dtype=np.float)

# 기본 히스토그램을 구한다.
for i in range(gray.shape[1]):
    for j in range(gray.shape[0]):
        gray_hist[gray.item(j,i)]+=1
# 정규화 시켜서 저장한다.
for i in range(256):
    norm_hist[i] = gray_hist[i] / (gray.shape[0]*gray.shape[1])

vwlist = []
#vwlist2 = [] #without weight

for i in range(1,256): #T값을 결정하기위해서 1부터~255단계까지 바꾸어 나간다
    w0 = 0.0
    w1 = 0.0
    u0 = 0.0
    u1 = 0.0
    v0 = 0.0
    v1 = 0.0
    for j in range(i):
        w0 += norm_hist[j] #T가 1이라면  정규화된 값을 넣고
    for j in range(i+1,256):
        w1 += norm_hist[j] # 나머지값(2부터255까지 누적값 더하기)을 w1에넣고
        
    if w0 != 0:
        for j in range(i):
            u0 += j*norm_hist[j]
        u0 /= w0
        for j in range(i):
            v0 += norm_hist[j]*(j-u0)**2
        v0 /= w0

    if w1 != 0:
        for j in range(i+1,256):
            u1 += j*norm_hist[j]
        u1 /= w1
        for j in range(i+1,256):
            v1 += norm_hist[j]*(j-u1)**2
        v1 /= w1

    v_within = w0 * v0 + w1 * v1
    #v_within2 = v0 + v1 #without weight
    vwlist.append(v_within)
    #vwlist2.append(v_within2) #without weight
    #if v_within < best:
        #best = v_within
        #best_t = i
        #T_max[0] = i
        #T_max[1] = v_within

#print(T_max)
#print(best, best_t)

#print(vwlist)
t_argmin = np.argmin(vwlist)
print(t_argmin, vwlist[t_argmin])
#argmin2 = np.argmin(vwlist2)
#print(argmin)
#plt.subplot(121)
plt.plot(vwlist)
#plt.subplot(122)
#plt.plot(vwlist2)
plt.show()

for i in range(gray.shape[0]):
    for j in range(gray.shape[1]):
        if gray[i,j] >= t_argmin:
            binary[i,j] = 255
        else:
            binary[i,j] = 0

cv.imshow('binary img',binary)
cv.waitKey(0)
cv.destroyAllWindows()
