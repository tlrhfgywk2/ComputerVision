import cv2
import numpy as np

# 체커보드 설정
chessboard_size = (5, 5)
square_size = 5 

# 카메라 내부 파라미터
K = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]]) 
dist_coef = np.zeros((4, 1))  # 왜곡 계수

# 객체 포인트 생성
objp = np.zeros((np.prod(chessboard_size), 3), dtype=np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

# 노트북 웹캡 On
cap = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = cap.read()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), 
                                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            _, rvec, tvec, _ = cv2.solvePnPRansac(objp, corners2, K, dist_coef)
            
            # 3D 큐브의 꼭지점 정의
            axis = np.float32([[square_size, 0, 0], [0, square_size, 0], [0, 0, -square_size]]).reshape(-1, 3)
            imgpts, _ = cv2.projectPoints(axis, rvec, tvec, K, dist_coef)
            
            # 체커보드의 첫 번째 코너를 시작점으로 설정
            imgpts = np.int32(imgpts).reshape(-1, 2)
            start_point = tuple(corners2[0].ravel())
            
            # 큐브의 꼭지점 그리기
            frame = cv2.line(frame, start_point, tuple(imgpts[0]), (255, 0, 0), 5)
            frame = cv2.line(frame, start_point, tuple(imgpts[1]), (0, 255, 0), 5)
            frame = cv2.line(frame, start_point, tuple(imgpts[2]), (0, 0, 255), 5)

            # 큐브의 엣지 그리기
            frame = cv2.line(frame, tuple(imgpts[0]), tuple(imgpts[1]), (255, 255, 0), 5)
            frame = cv2.line(frame, tuple(imgpts[1]), tuple(imgpts[2]), (255, 255, 0), 5)
            frame = cv2.line(frame, tuple(imgpts[2]), tuple(imgpts[0]), (255, 255, 0), 5)

            cv2.drawChessboardCorners(frame, chessboard_size, corners2, ret)
        
        cv2.imshow("3D Cube", frame)
        
        # q를 누르면 웹캠 꺼짐
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
