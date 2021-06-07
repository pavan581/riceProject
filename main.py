import cv2
from imutils import contours
import imutils


#reading image
image = cv2.imread("25.jpeg")
cv2.imshow("image", image)

#converting to gray scale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray_image", gray_image)

#applying canny edge detection
edge = cv2.Canny(gray_image, 10, 250)
cv2.imshow("edge", edge)

# Find contours
cnts = cv2.findContours(edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# Sort contours from left to right as leftmost contour is reference object
#(cnts, _) = contours.sort_contours(cnts)

# Remove contours which are not large enough
cnts = [x for x in cnts if cv2.contourArea(x) > 100]

idx = 0
for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    
    if w>50 and h>50:
        idx+=1
        new_img=image[y:y+h, x:x+w]
        cv2.imshow("new_img", new_img)

        #cropping images
        cv2.imwrite("cropped/"+str(idx) + '.png', new_img)

print("...")
