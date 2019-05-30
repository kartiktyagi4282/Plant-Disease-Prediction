import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread("66661.png")
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
stencil = np.zeros(img.shape).astype(img.dtype)
c=[255,0,0]
indices = np.where(np.all(img == c, axis=-1))
a=indices[1]
b=indices[0]
x_mn=min(a)
x_mx=max(a)
y_mn=min(b)
y_mx=max(b)
#print(x_mn,x_mx)
#print(y_mn,y_mx)
img_org = cv2.imread("6666.png")
img_org=cv2.cvtColor(img_org,cv2.COLOR_BGR2RGB)
indices=np.concatenate((np.expand_dims(a,1),np.expand_dims(b,1)),axis=1)
contours=[indices]
stencil = np.zeros(img.shape).astype(img.dtype)
img2=cv2.drawContours(stencil,contours,-1,(255,255,255),thickness=cv2.FILLED)
result = cv2.bitwise_and(img_org,img2)
result=result[y_mn:y_mx,x_mn:x_mx]
plt.imshow(result)
cv2.imwrite("res.png", result)