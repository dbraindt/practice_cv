import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
#%%

#img = cv2.imread('batman.j resxtpg')
#plt.imshow(img)
#
##img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
##plt.imshow(img_rgb)
#
#img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#plt.imshow(img_gray, cmap = 'gray')

#%%

img = np.asarray(Image.open("data/batman.jpg"))
#img = np.asarray(Image.open("data/batman3.jpg"))
img2 = np.mean(img, axis=2)
plt.imshow(img2, cmap='gray')
plt.show()
#%%

Rlinear = img[:,:,0]
Glinear = img[:,:,1] 
Blinear = img[:,:,2]
Ylinear = 0.2126 * Rlinear + 0.7152 * Glinear + 0.0722 * Blinear
#%%
plt.imshow(Ylinear)#, cmap='gray')
plt.show()
#%%
plt.savefig('somegrayscale.png')
