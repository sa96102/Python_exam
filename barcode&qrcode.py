import pyzbar.pyzbar as pyzbar
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('Data/barcode.jpg')

plt.imshow(img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.imshow(gray, cmap='gray')

decoded = pyzbar.decode(gray)

decoded

for d in decoded:
    print(d.data.decode('utf-8'))
    print(d.type)
    
    cv2.rectangle(img, (d.rect[0], d.rect[1]), (d.rect[0] + d.rect[2], d.rect[1] + d.rect[3]), (0, 0, 255), 2)
    
plt.imshow(img)