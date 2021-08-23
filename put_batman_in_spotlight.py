import numpy as np
import cv2 as cv

# run, fools


#%%
# lets dream a bit
# take batman
# grayscale, shrink
# contrast
# mask
# detect circle
# make a patch of the right size
# put batman in a patch
# compose two images
# np.clip
# new_image = cv.convertScaleAbs(image, alpha=alpha, beta=beta)
# fix brightness(beta?), contrast(alpha?)
# exposure(gamma?) correction?
"""
lookUpTable = np.empty((1,256), np.uint8)
for i in range(256):
    lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
res = cv.LUT(img_original, lookUpTable)
"""

#%%

batman = cv.imread('data/batman.jpg', cv.IMREAD_COLOR)
spotlight = cv.imread('data/spotlight.png', cv.IMREAD_COLOR)

#%%

cv.imshow('batman', batman)
cv.imshow('spotlight', spotlight)
cv.waitKey(0)
cv.destroyAllWindows()
#%%

# 250 250 3
#spotlight.shape

# 316 474 3
#batman.shape

#%%

batman_8u = cv.cvtColor(batman, cv.CV_8U)
cv.imshow('batshit', batman_8u)
cv.waitKey(0)
cv.destroyAllWindows()
#%%

# shrink - INTER_AREA
# enlarge - INTER_CUBIC or INTER_LINEAR
small_batman = cv.resize(batman, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)

cv.imshow('small_batman', small_batman)
cv.waitKey(0)
cv.destroyAllWindows()
#%%





cv.imshow('spot', spotlight[0:200, 0:200])
cv.imshow('thing', batman[0:200,0:200])
cv.waitKey(0)
cv.destroyAllWindows()
#%%

# dimensions must match
q = cv.addWeighted(spotlight[0:200, 0:200], 0.5, batman[0:200,0:200], 0.5, 0.0)

cv.imshow('thing', q)
cv.waitKey(0)
cv.destroyAllWindows()
#%%

batheight, batwidth = batman.shape[:2]
spotheight, spotwidth = spotlight.shape[:2]

# scaling
#cv.resize(spotlight, (1.2*spotheight, 1.2*spotwidth), interpolation=cv.INTER_CUBIC)

# translation
M = np.float32([[1,0,100],[0,1,50]]) # translate by +100, +50

shiftman = cv.warpAffine(batman, M, (batwidth +100, batheight+50))


M2 = cv.getRotationMatrix2D(((batwidth-1)/2, (batheight -1 )/2), -20, 1)
rotman = cv.warpAffine(batman, M2, (batwidth, batheight))


cv.imshow('rotman', rotman)
cv.waitKey(0)
cv.destroyAllWindows()
# Affine transformation - all parallel lines in the original image will remain parallel in the output image
# (non-parallel lines will get all sorts of distorted)

#%%
# affine transformation from two sets of three points

pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100, 210]])

M3 = cv.getAffineTransform(pts1, pts2)

a_fine_man = cv.warpAffine(batman, M3, (batwidth, batheight))


cv.imshow('batman', batman)
cv.imshow('affineman', a_fine_man)
cv.waitKey(0)
cv.destroyAllWindows()

#%%
# perspective transformation
# straight lines will remain straight
# parallel lines out the window

#need 4 points on the input image
#3 of them should not be collinear


pts1 = np.float32([[50,50],[200,50],[50,200], [200,200]])
pts2 = np.float32([[60,100],[180,50],[100, 210], [200,200]])

M4 = cv.getPerspectiveTransform(pts1, pts2)

a_perspective_man = cv.warpPerspective(batman, M4, (batwidth, batheight))


cv.imshow('batman', batman)
cv.imshow('perspectiveman', a_perspective_man)
cv.waitKey(0)
cv.destroyAllWindows()

#%%

def extract_spotlight_geometry(spotlight):
    gray = cv.cvtColor(spotlight, cv.COLOR_BGR2GRAY)
    grayboost = cv.convertScaleAbs(gray, alpha=1.1, beta=-1)
    ret, mask = cv.threshold(grayboost, thresh=220, maxval=255, type=cv.THRESH_BINARY)
    return mask, gray, grayboost

mask, gray, grayboost = extract_spotlight_geometry(spotlight)

mask_inv = cv.bitwise_not(mask)

cv.imshow('spotlight_mask', mask)
cv.imshow('spotlight_mask_inv', mask_inv)
#cv.imshow('spotlight_gray', gray)
#cv.imshow('spotlight_grayboost', grayboost)
cv.waitKey(0)
cv.destroyAllWindows()
#%%


min_x = 0
min_y = 0

max_x = mask.shape[1] - 1
max_y = mask.shape[0] - 1

# find min x and max x
for row in range(mask.shape[0]):
    if mask[row,:].argmax() > 0:
        min_row = row
        break

for row in range(mask.shape[0]-1, 0, -1):
    if mask[row,:].argmax() > 0:
        max_row = row
        break

for col in range(mask.shape[1]):
    if mask[:,col].argmax() > 0:
        min_col = col
        break

for col in range(mask.shape[1] -1, 0, -1):
    if mask[:,col].argmax() > 0:
        max_col = col
        break

#%%
# unless im mistaken
# (column, row)
# (x down, y down)

q = cv.rectangle(mask, pt1=( min_col, min_row), pt2=(max_col, max_row ), color=(200,200,200), lineType=cv.LINE_8)

cv.imshow('spotlight_mask', mask)
cv.imshow('spotlight_grayboost', grayboost)
cv.imshow('some_contour', q)
cv.waitKey(0)
cv.destroyAllWindows()

#%%

# this is where batman will go
center_x = int((max_col - min_col + 1) / 2 + min_col)
center_y = int((max_row - min_row + 1 ) / 2 + min_row)
radius = int((max_col - min_col + 1) / 2)
print(center_x, center_y, radius)

# so is it row-col or col row?
pt1 = (min_col, center_y)
pt2 = (max_col, center_y)
pt3 = (center_x, center_y-radius)
pt3p = (center_x, min_row)


cv.circle(mask_inv, center=(center_x, center_y), radius = radius, color=(200,220,200))
cv.polylines(mask_inv, pts=[np.array([pt1,pt2,pt3])], isClosed=True, color=(150,100,150))
cv.polylines(mask_inv, pts=[np.array([pt1,pt2,pt3p])], isClosed=True, color=(150,100,150))

cv.imshow('some_contour', mask_inv)
cv.waitKey(0)
cv.destroyAllWindows()

#%%

Mtilt = cv.getAffineTransform(np.float32([pt1,pt2,pt3]),np.float32([pt1,pt2,pt3p]))

Mrotate = cv.getRotationMatrix2D(((batwidth-1)/2, (batheight -1 )/2), -30, 1)
rotated_batman = cv.warpAffine(batman, Mrotate, (batwidth, batheight))
tilted_batman = cv.warpAffine(rotated_batman, Mtilt, (batwidth, batheight))
small_batman = cv.resize(tilted_batman, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)

cv.imshow('batman', batman)
cv.imshow('rotated_batman', rotated_batman)
cv.imshow('tilted_batman', tilted_batman)
cv.imshow('small_batman', small_batman)
# TODO apply mask here
#
cv.waitKey(0)
cv.destroyAllWindows()

#%%

def extract_batman(batman):
    gray = cv.cvtColor(batman, cv.COLOR_BGR2GRAY)
    grayboost = cv.convertScaleAbs(gray, alpha=1.1, beta=-1)
    ret, mask = cv.threshold(grayboost, thresh=220, maxval=255, type=cv.THRESH_BINARY)
    return mask, gray, grayboost

batmask, graybat, batboost = extract_batman(batman)
batmask_inv = cv.bitwise_not(batmask)
cv.imshow('spotlight_mask', batmask)
cv.imshow('spotlight_mask_inv', batmask_inv)
cv.imshow('spotlight_gray', graybat)
cv.imshow('spotlight_grayboost', batboost)
cv.waitKey(0)
cv.destroyAllWindows()
#%%
rotated_batmask = cv.warpAffine(batmask_inv, Mrotate, (batwidth, batheight))
tilted_batmask = cv.warpAffine(rotated_batmask, Mtilt, (batwidth, batheight))
small_batmask = cv.resize(tilted_batmask, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)

cv.imshow('batman', batmask_inv)
cv.imshow('rotated_batman', rotated_batmask)
cv.imshow('tilted_batman', tilted_batmask)
cv.imshow('small_batman', small_batmask)
cv.waitKey(0)
cv.destroyAllWindows()
#%%

_height, _width, _n_channels = spotlight.shape
result = np.zeros(spotlight.shape, spotlight.dtype)

x = cv.bitwise_and(small_batman, small_batman, mask=small_batmask)
# take size of small_batmask, center around center_x, center_y, yank a chunk of
q = spotlight[slice(min_row, max_row),slice(min_col, max_col), :]


batheight, batwidth, batchannels = x.shape
batcenter_x = int(batwidth / 2) + 1
batcenter_y = int(batheight / 2) + 1

roi_height, roi_width, nchannels = q.shape
print(roi_height)
print(roi_width)

sliced = x[
    slice(int(batcenter_y-roi_height/2), int(batcenter_y+roi_height/2)),
    slice(int(batcenter_x-roi_width/2), int(batcenter_x+roi_width/2)), :]

sliced_batmask = small_batmask[
         slice(int(batcenter_y-roi_height/2), int(batcenter_y+roi_height/2)),
         slice(int(batcenter_x-roi_width/2), int(batcenter_x+roi_width/2))]

#ppp = cv.add(q, q, mask=cv.bitwise_not(sliced_batmask))
#qqq = cv.add(q, sliced, mask=sliced_batmask)

negatively_background = cv.bitwise_and(q,q,
                                       cv.bitwise_not(sliced_batmask))
positively_batman = cv.bitwise_and(sliced,sliced,
                                   sliced_batmask)

patch = cv.add(negatively_background, positively_batman, mask=cv.bitwise_not(sliced_batmask))

result = spotlight
result[slice(min_row, max_row),slice(min_col, max_col), :] = patch
#cv.imshow('a', q)
#cv.imshow('b',sliced)
cv.imshow('c',positively_batman)
cv.imshow('d',negatively_background)
cv.imshow('e',patch)
cv.imshow('finally', result)
cv.waitKey(0)
cv.destroyAllWindows()
#
#Mtranslate = np.array([
#    [1, 0, center_x],
#    [0, 1, center_y], np.float64])
#out = cv.transform(x, Mtranslate, (_height, _width))
