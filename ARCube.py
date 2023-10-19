import numpy as np
import cv2
import glob

def calibrate_camera(chessboard_size, square_size, images_path):
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []
    imgpoints = []

    images = glob.glob(images_path)

    gray = None

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 체스 보드 코너를 찾자
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            imgpoints.append(corners2)
            img = cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(200)

    cv2.destroyAllWindows()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return ret, mtx, dist

def project_points(points_3d, rvecs, tvecs, camera_matrix):
    R, _ = cv2.Rodrigues(rvecs) # 회전 벡터 -> 회전 행렬
    RT = np.concatenate((R, tvecs), axis=1) # 외부 매개변수 행렬
    P = np.matmul(camera_matrix, RT) # 투영 행렬
    
    points_3d_homogeneous = np.concatenate((points_3d, np.ones((points_3d.shape[0], 1))), axis=1) # 동차 좌표로 변환
    points_2d = np.matmul(P, points_3d_homogeneous.T).T # 평면에 투영

    # 정규화한다
    for i in range(points_2d.shape[0]):
        points_2d[i, 0] /= points_2d[i, 2]
        points_2d[i, 1] /= points_2d[i, 2]
        points_2d[i, 2] = 1

    return points_2d[:, :2]

def draw_cube(img, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    color = (255, 0, 0)
    lines = [(0,1), (1,2), (2,3), (3,0), (0,4), (1,5), (2,6), (3,7), (4,5), (5,6), (6,7), (7,4)]

    for i, j in lines:
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color, 3)

    return img

# 카메라 캘리브레이션 설정
# camera_index = cv2.CAP_DSHOW # 노트북 웹캠
camera_index = cv2.CAP_DSHOW + 1 # USB 웹캠
chessboard_size = (4,4)
square_size = 1.0
calibration_images_path = 'calibration_images/*.jpg'  # 캘리브레이션 이미지 경로

ret, mtx, dist = calibrate_camera(chessboard_size, square_size, calibration_images_path)

# 웹캠 준비
camera = cv2.VideoCapture(camera_index)

while(True):
    # 캡처
    ret, frame = camera.read()

    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 체스보드 코너 찾기
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret == True:
            objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
            objp *= square_size

            # 큐브 꼭지점
            axis = np.float32([[0,0,0], [1,0,0], [1,1,0], [0,1,0],
                               [0,0,-1],[1,0,-1],[1,1,-1],[0,1,-1]])

            # 회전 및 변환 벡터를 계산
            _, rvecs, tvecs = cv2.solvePnP(objp, corners, mtx, dist)

            # 큐브의 꼭지점을 평면에 투영
            imgpts = project_points(axis, rvecs, tvecs, mtx)

            frame = draw_cube(frame, imgpts)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

camera.release()
cv2.destroyAllWindows()