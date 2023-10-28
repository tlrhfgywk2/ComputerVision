import cv2
import numpy as np

def erp2rect(src, theta=np.deg2rad(180), hfov=np.deg2rad(150), vfov=np.deg2rad(150)):
    src_rows, src_cols, src_channels = src.shape

    f = src_cols / (2 * np.pi)
    dst_cols = int(2 * f * np.tan(hfov / 2) + 0.5)
    dst_rows = int(2 * f * np.tan(vfov / 2) + 0.5)
    dst = np.zeros((dst_rows, dst_cols, src_channels), dtype=src.dtype)

    dst_cx = dst_cols / 2
    dst_cy = dst_rows / 2

    for x in range(dst_cols):
        xth = np.arctan((x - dst_cx) / f)
        src_x = int((xth + theta) * src_rows / np.pi + 0.5)
        
        yf = f / np.cos(xth)
        for y in range(dst_rows):
            yth = np.arctan((y - dst_cy) / yf)
            src_y = int(yth * src_rows / np.pi + src_rows / 2 + 0.5)
            dst[y, x] = src[src_y, src_x]

    return dst

src_image = cv2.imread("erp.png")
rect_image = erp2rect(src_image)
cv2.imwrite("front_view_150_150.jpg", rect_image)
