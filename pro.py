from scipy.spatial.distance import euclidean
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2
import math

#Varaibles to store data
height=list()
width=list()
Area=list()
Perimeter=list()

# Function to show array of images (intermediate results)
def show_images(images):
	for i, img in enumerate(images):
		cv2.imshow("image_" + str(i), img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

#Function to resize viewing window
def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


#img_path = "images/example_02.jpg"

# Read image and preprocess
image = cv2.imread('jaya1.jpg')
cv2.imshow('input image', ResizeWithAspectRatio(image, width=550))

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    #Convert to Gray Scale
blur = cv2.GaussianBlur(gray, (9, 9), 0)          #Gaussian Blur filter

edged = cv2.Canny(blur, 50, 100)                  #Canny Edge algorithm
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

#show_images([blur, edged])

# Find contours
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# Sort contours from left to right as leftmost contour is reference object
(cnts, _) = contours.sort_contours(cnts)

# Remove contours which are not large enough
cnts = [x for x in cnts if cv2.contourArea(x) > 100]

#cv2.drawContours(image, cnts, -1, (0,255,0), 3)
#show_images([image, edged])
#print(len(cnts))

# Reference object dimensions
ref_object = cnts[0]
box = cv2.minAreaRect(ref_object)
box = cv2.boxPoints(box)
box = np.array(box, dtype="int")
box = perspective.order_points(box)
(tl, tr, br, bl) = box
dist_in_pixel = euclidean(tl, tr)
dist_in_cm = 0.87
pixel_per_cm = dist_in_pixel/dist_in_cm

# Draw remaining contours
for cnt in cnts:
	box = cv2.minAreaRect(cnt)
	box = cv2.boxPoints(box)
	box = np.array(box, dtype="int")
	box = perspective.order_points(box)
	(tl, tr, br, bl) = box
	cv2.drawContours(image, [box.astype("int")], -1, (0, 0, 255), 2)
	mid_pt_horizontal = (tl[0] + int(abs(tr[0] - tl[0])/2), tl[1] + int(abs(tr[1] - tl[1])/2))
	mid_pt_verticle = (tr[0] + int(abs(tr[0] - br[0])/2), tr[1] + int(abs(tr[1] - br[1])/2))
	wid = euclidean(tl, tr)/pixel_per_cm
	ht = euclidean(tr, br)/pixel_per_cm
	perimeter=cv2.arcLength(cnt,True)/pixel_per_cm
	area=cv2.contourArea(cnt)/(pixel_per_cm*pixel_per_cm)
	
	height.append(ht)
	width.append(wid)
	Area.append(area)
	Perimeter.append(perimeter)
	cv2.putText(image, "{:.1f}cm".format(wid), (int(mid_pt_horizontal[0] - 15), int(mid_pt_horizontal[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
	cv2.putText(image, "{:.1f}cm".format(ht), (int(mid_pt_verticle[0] + 10), int(mid_pt_verticle[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

cv2.imshow('image', ResizeWithAspectRatio(image, width=550))

#Calculating other parameters
#Aspect Ratio
ar=[width[i]/height[i] for i in range(len(height))]
#Roughness
roughness=[(Perimeter[i]*Perimeter[i])/(4*3.14*Area[i]) for i in range(len(Perimeter))]
#Metric
metric=[1/roughness[i] for i in range(len(roughness))]
#Area of enclosing rectangle
a_en_rec=[width[i]*height[i] for i in range(len(height))]
#Solidity
solidity = [Area[i]/a_en_rec[i] for i in range(len(Area))]

print("Grain : Height : Width : Aspect Ratio : Area : Roughness : Metric : Solidity")
for i in range(len(height)):
        print("grain[{}]:{}:{}:{}:{}:{}:{}:{}".format(i+1, height[i], width[i], ar[i], Area[i], roughness[i], metric[i], solidity[i]))
