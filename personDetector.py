import ultralytics
import cv2
import numpy as np
from ultralytics import YOLO

imx_model = YOLO("yolo11n_imx_model")

results = imx_model("https://ultralytics.com/images/bus.jpg")

cv2.imshow("img", results[0].plot())
cv2.waitKey(q)
cv2.destroyAllWindows()

