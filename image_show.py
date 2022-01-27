# Python program to explain cv2.imshow() method

# importing cv2
import cv2

# path
path = 'image.jpg'

# Reading an image in default mode
image = cv2.imread(path)

# Window name in which image is displayed
window_name = 'image'

cv2.imwrite('rewrite.jpg', image)
