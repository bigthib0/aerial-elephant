import numpy as np
import cv2

arr = np.random.rand(256, 256)
arr = (arr * 255).astype(np.uint8)

print(arr.min())
print(arr.max())


cv2.imwrite('test.png', arr)