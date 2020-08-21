import cv2
import numpy as np

img = cv2.imread("input.jpg")
img = cv2.resize(img, (1300, 800))
original_copy = img.copy()


gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_img = cv2.GaussianBlur(gray_img, (7,11), cv2.BORDER_DEFAULT)
canny_img = cv2.Canny(gray_img, 0, 120)
contours, hierarchy= cv2.findContours(canny_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)


for c in contours:
    tmp = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02*tmp, True)
    print(len(approx))
    if len(approx) == 4:
        result = approx
        break


try:
    result = result.reshape((4,2))
except:
    print("Cannot find the document")
    exit()

result_temp = np.zeros((4,2), dtype = np.float32)

sum_ = result.sum(axis=1)
result_temp[0] = result[np.argmin(sum_)]
result_temp[2] = result[np.argmax(sum_)]

diff_ = np.diff(result, axis=1)
result_temp[1] = result[np.argmin(diff_)]
result_temp[3] = result[np.argmax(diff_)]

pts = np.float32([[0, 0], [800, 0], [800, 800], [0, 800]])
perspective = cv2.getPerspectiveTransform(result_temp, pts)
final = cv2.warpPerspective(original_copy, perspective, (800, 800))

while True:
    cv2.imshow("Out",final)
    if cv2.waitKey(1) == 27:
        break