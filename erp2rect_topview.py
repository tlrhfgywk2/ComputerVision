import cv2
import numpy as np

def erp2rect(src, hfov=np.deg2rad(120), vfov=np.deg2rad(120)):
    H, W, src_channels = src.shape

    f = W / (2 * np.pi)
    W_ = int(2 * f * np.tan(hfov / 2) + 0.5)
    H_ = int(2 * f * np.tan(vfov / 2) + 0.5)
    dst = np.zeros((H_, W_, src_channels), dtype=src.dtype)

    cx = int(W / 2)
    cy = int(H / 2)
    cx_ = int(W_ / 2)
    cy_ = int(H_ / 2)

    for x in range(W_):
        for y in range(H_):
            theta = abs(x - cx) * 2 * np.pi / W
            phi = (cy - y) * np.pi / H
            D = int(f / np.tan(phi))

            x_ = int(cx_ + D * np.sin(theta))
            y_ = int(cy_ - D * np.cos(theta))

            print(f"[{x}, {y}]", D, np.rad2deg(phi), np.rad2deg(theta), x_, y_)
            dst[y, x] = src[y_, x_]

    return dst

src_image = cv2.imread("erp.png")
rect_image = erp2rect(src_image)
cv2.imwrite("topview_120_120.jpg", rect_image)
