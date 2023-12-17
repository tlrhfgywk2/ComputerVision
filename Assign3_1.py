import os
import sys
import numpy as np
import cv2
import glob

# 카메라 캘리브레이션
def calibrate_camera(checkerboard_size, images_path):
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)

    # 각 이미지에 대한 3D 포인트 및 이미지 포인트를 저장할 배열
    objpoints = [] # 3d 포인트
    imgpoints = [] # 2d 포인트

    # 체스보드 이미지 로드
    images = glob.glob(images_path)
    corner_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for fname in images:
        print("이미지 :", fname)
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 체스보드 코너 찾기
        isFind, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

        # 코너 찾은 경우
        if isFind:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), corner_criteria)
            imgpoints.append(corners2)
    
    # cv2.destroyAllWindows()
    # 카메라 캘리브레이션
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return ret, mtx, dist

def android_camera():
    video = cv2.VideoCapture(1) # 0은 노트북 웹캠

    width = int(video.get(3)) # 가로 길이 가져오기 
    height = int(video.get(4)) # 세로 길이 가져오기

    # video.set(cv2.CAP_PROP_FPS, 60)
    fps = video.get(cv2.CAP_PROP_FPS)
    print(f"W : {width}, H : {height}, fps : {fps}")

    return video

# 체스보드 코너 설정
checkerboard_size = (4, 4)
ret, mtx, dist = calibrate_camera(checkerboard_size, 'Images/*.png')
dist = np.zeros(5)

REF_DIR = f'Reference/'
if not os.path.exists(REF_DIR):
    os.makedirs(REF_DIR)

video = android_camera()

# 레퍼런스 영상 저장
while True:
    isValid, frame = video.read()
    if not isValid:
        print("Fail to read frame!")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', frame)

    key = cv2.waitKey(1)
    if key == ord(' '):
        # 체스보드 코너 찾기
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
        if ret == True:
            objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)

            # 회전 및 변환 벡터를 계산
            _, tmp_rvec, tmp_tvec = cv2.solvePnP(objp, corners, mtx, dist)

            cv2.imwrite(f'{REF_DIR}ref.png', frame)
            
            # DB에 저장
            np.savez(f'{REF_DIR}ref_pose', ref_rvec=tmp_rvec, ref_tvec=tmp_tvec)
        break
    elif key == ord('q'):
        break

def triangulate_points(P1, P2, pts1, pts2):
    num_points = pts1.shape[0]
    X = np.zeros((num_points, 4))

    for i in range(num_points):
        A = np.zeros((4, 4))
        A[0] = pts1[i, 0] * P1[2] - P1[0]
        A[1] = pts1[i, 1] * P1[2] - P1[1]
        A[2] = pts2[i, 0] * P2[2] - P2[0]
        A[3] = pts2[i, 1] * P2[2] - P2[1]

        # 특이값 분해를 통해 최소 고유 벡터를 찾는다
        _, _, V = np.linalg.svd(A)
        X[i] = V[-1, :]

    X /= X[:, 3][:, np.newaxis]

    return X[:, :3]

# 투영 오차 계산
def calculate_reprojection_error(features_worldaxis, img_pts, rvec, tvec, mtx, dist):
    # img_pts_reprojected, _ = cv2.projectPoints(features_worldaxis, rvec, tvec, mtx, dist)
    img_pts_reprojected = project_points(features_worldaxis, rvec, tvec, mtx)
    img_pts_reprojected = img_pts_reprojected.reshape(-1, 2).astype(np.float32)
    img_pts = img_pts.astype(np.float32)
    errors = np.sqrt(np.sum((img_pts - img_pts_reprojected)**2, axis=1))
    return errors

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

# 투영 및 렌더링
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

video.release()
cv2.destroyAllWindows()

min_inlier_num = 100

fdetector = cv2.ORB_create()
fmatcher = cv2.DescriptorMatcher_create('BruteForce-Hamming')

obj_image = cv2.imread(f'{REF_DIR}ref.png')
assert obj_image is not None
obj_keypoints, obj_descriptors = fdetector.detectAndCompute(obj_image, None)
assert len(obj_keypoints) >= min_inlier_num
fmatcher.add(obj_descriptors)

video = android_camera()
points = []
fps_count = 0
data_count = 0
save_count = 0

if os.path.exists(f'{REF_DIR}features_worldaxis.npz'):
    os.remove(f'{REF_DIR}features_worldaxis.npz')

while True:
    isValid, img = video.read()
    if not isValid:
        break

    img_keypoints, img_descriptors = fdetector.detectAndCompute(img, None)
    match = fmatcher.match(img_descriptors, obj_descriptors)
    if len(match) < min_inlier_num:
        continue

    obj_pts, img_pts = [], []
    for m in match[:100]:
        obj_pts.append(obj_keypoints[m.trainIdx].pt)
        img_pts.append(img_keypoints[m.queryIdx].pt)
    obj_pts = np.array(obj_pts, dtype=np.float32)
    obj_pts = np.hstack((obj_pts, np.zeros((len(obj_pts), 1), dtype=np.float32))) # 2D -> 3D
    img_pts = np.array(img_pts, dtype=np.float32)

    _, _, _, inliers = cv2.solvePnPRansac(obj_pts, img_pts, mtx, dist, useExtrinsicGuess=False,
                                                 iterationsCount=500, reprojectionError=2., confidence=0.99)

    inlier_mask = np.zeros(len(match), dtype=np.uint8)
    inlier_mask[inliers] = 1
    img_result = cv2.drawMatches(img, img_keypoints, obj_image, obj_keypoints, match, None, (0, 0, 255), (0, 127, 0), inlier_mask)
    
    cv2.putText(img_result, f"Save Count {save_count} | Data Count {data_count}", (10, 25), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))
    cv2.imshow('Image', img_result)
    fps_count += 1

    # 3프레임마다 실행 (1초 동안 10번 실행)
    if fps_count % 3 == 0:
        fps_count = 0
        save_count += 1
        try:
            obj_pts = obj_pts[:, :2]

            # DB에 저장되어 있던 체커보드-레퍼런스 영상 [R|t] 로드
            ref_pose = np.load(f'{REF_DIR}ref_pose.npz')
            ref_rvec = ref_pose['ref_rvec']
            ref_tvec = ref_pose['ref_tvec']

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            chess_ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
            
            if chess_ret:
                objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
                objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)

                # 원점은 체커보드
                # 레퍼런스 카메라와의 [R|t]
                ref_R, _ = cv2.Rodrigues(ref_rvec)

                # 실시간 카메라와의 [R|t]
                _, cur_rvec, cur_tvec = cv2.solvePnP(objp, corners, mtx, dist)
                cur_R, _ = cv2.Rodrigues(cur_rvec)

                P0 = mtx @ np.hstack((ref_R, ref_tvec))
                P1 = mtx @ np.hstack((cur_R, cur_tvec))


                # 1. 라이브러리 사용
                # features_worldaxis = cv2.triangulatePoints(P0, P1, obj_pts.T, img_pts.T)
                # features_worldaxis /= features_worldaxis[3]
                # features_worldaxis = features_worldaxis.T
                # features_worldaxis = features_worldaxis[:, :3]

                # 2. 직접 구현
                features_worldaxis = triangulate_points(P0, P1, obj_pts, img_pts)

                # 월드 좌표 검증
                # 투영 오차를 계산하여 출력
                if len(features_worldaxis) > 0:
                    reprojection_errors = calculate_reprojection_error(features_worldaxis, img_pts, cur_rvec, cur_tvec, mtx, dist)
                    threshold = 0.25

                    # 임계값보다 큰 투영 오차를 가진 특징점 제외
                    mask = reprojection_errors < threshold
                    featuers_worldaxis_inliers = features_worldaxis[mask]
                    img_pts_inliers = img_pts[mask]

                    print(f"Number of inliers: {len(featuers_worldaxis_inliers)}")
                    features_worldaxis = featuers_worldaxis_inliers

                    # DB에 저장
                    try:
                        data = np.load(f'{REF_DIR}features_worldaxis.npz')
                        saved_worldaxis = data['worldaxis']
                    except FileNotFoundError:
                        saved_worldaxis = np.array([])

                    if saved_worldaxis.size == 0:
                        merged_worldaxis = features_worldaxis
                    else:
                        merged_worldaxis = np.vstack([saved_worldaxis, features_worldaxis])

                    # 새로 계산된 데이터와 기존 데이터를 합치고 중복 제거
                    # 소수점 6자리까지 동일한 월드 좌표는 제거
                    unique_indices = np.unique(np.round(merged_worldaxis, decimals=6), axis=0, return_index=True)[1]

                    # 중복 제거한 데이터 저장
                    unique_worldaxis = merged_worldaxis[unique_indices]
                    np.savez(f'{REF_DIR}features_worldaxis', worldaxis=unique_worldaxis)
                    data = np.load(f'{REF_DIR}features_worldaxis.npz')
                    print(f"Saved {len(data['worldaxis'])}")
                    data_count = len(data['worldaxis'])
        except Exception as e:
            print(e)
            import traceback
            print(traceback.format_exc())
            cv2.destroyAllWindows()
            sys.exit()

    key = cv2.waitKey(1)
    if key == ord(' '):
        break
    if key == ord('q'):
        video.release()
        cv2.destroyAllWindows()
        sys.exit()

data = np.load(f'{REF_DIR}features_worldaxis.npz')
features_worldaxis = data['worldaxis']

while True:
    if len(features_worldaxis) == 0:
        break

    isValid, img = video.read()
    if not isValid:
        break

    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        isFind, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
        if isFind:
            objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
            # 회전 및 변환 벡터를 계산
            _, rvec, tvec = cv2.solvePnP(objp, corners, mtx, dist)

            # impts, _ = cv2.projectPoints(X, rvec, tvec, mtx, dist)
            impts = project_points(features_worldaxis, rvec, tvec, mtx)
            impts = np.int32(impts).reshape(-1,2)

            impts = impts.squeeze()
            for p in impts:
                if p[0] > 640 or p[1] > 480 or p[0] < 0 or p[1] < 0:
                    continue
                cv2.circle(img, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

        cv2.putText(img, f"Features Count : {len(features_worldaxis)}", (10, 25), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))
        cv2.imshow('Image', img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    except Exception as e:
        print(e)
        import traceback
        print(traceback.format_exc())
        break

video.release()
cv2.destroyAllWindows()