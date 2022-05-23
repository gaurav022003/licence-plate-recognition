import cv2
import numpy as np
import imutils
import easyocr
def number(img,gray,location):
    blank = np.zeros(gray.shape, dtype='uint8')
    mask = cv2.drawContours(blank, [location], 0, 255, -1)
    plate = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow('plate', plate)
    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    plate = gray[x1:x2 + 1, y1:y2 + 1]
    cv2.imshow('croped',plate)
    reader = easyocr.Reader(['en'])
    number = reader.readtext(plate)
    return (number)


def recognize(img):
    img = cv2.imread(img)
    cv2.imshow('original', img)
    gray=cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # cv2.imshow("gray", gray)
    filter = cv2.bilateralFilter(gray, 10, 10, 10)
    # cv2.imshow("filter", filter)
    edges=cv2.Canny(filter, 200, 250)
    # cv2.imshow('edges',edges)
    points=cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours= imutils.grab_contours(points)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    size=0
    found= 0
    for contour in contours:
        rect=cv2.approxPolyDP(contour, 10, True)
        if len(rect)==4:
            plate=number(img,gray,rect)
            size = len(plate)
            found=1
            break
    if found == 0:
        return "not found"
    else:
        return (plate[size - 1][1])
    # cv2.waitKey(0)
