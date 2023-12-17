import os
import numpy as np
import cv2
import sys
import glob

def android_camera():
    video = cv2.VideoCapture(1) # 0은 노트북 웹캠

    width = int(video.get(3)) # 가로 길이 가져오기 
    height = int(video.get(4)) # 세로 길이 가져오기

    # video.set(cv2.CAP_PROP_FPS, 60)
    fps = video.get(cv2.CAP_PROP_FPS)
    print(f"W : {width}, H : {height}, fps : {fps}")

    return video

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
    img_pts = img_pts.astype(np.float32)  # 데이터 타입을 float32로 변환
    # error = cv2.norm(img_pts, img_pts_reprojected, cv2.NORM_L2) / len(img_pts_reprojected)
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

save_count = 0

f, cx, cy = 296., 320., 240.
mtx = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
dist = np.zeros(5)

REF_DIR = f'Reference3_2/'
if not os.path.exists(REF_DIR):
    os.makedirs(REF_DIR)

video = android_camera()

while True:
    isValid, img = video.read()
    if not isValid:
        print("Fail to read frame!")
        break
    
    cv2.putText(img, f"Save Count {save_count}", (10, 25), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))
    cv2.imshow('Image', img)

    key = cv2.waitKey(1)
    if key == ord(' '):
        cv2.imwrite(f'{REF_DIR}Image_{save_count}.png', img)
        save_count += 1
        if save_count == 4:
            break
    elif key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

video = android_camera()
points = []
# fps_count = 0
data_count = 0

if os.path.exists(f'{REF_DIR}features_worldaxis.npz'):
    os.remove(f'{REF_DIR}features_worldaxis.npz')

min_inlier_num = 1

fdetector = cv2.ORB_create()

# 첫 번째 영상을 원점으로 설정
file_images = glob.glob(f'{REF_DIR}Image_*.png')
obj_images = [] 
obj_detector_pairs = []
fmatchers = []
for _image in file_images:
    fmatcher = cv2.DescriptorMatcher_create('BruteForce-Hamming')
    obj_image = cv2.imread(_image)
    assert obj_image is not None
    obj_keypoints, obj_descriptors = fdetector.detectAndCompute(obj_image, None)
    obj_detector_pairs.append((obj_keypoints, obj_descriptors))
    assert len(obj_keypoints) >= min_inlier_num
    fmatcher.add(obj_descriptors)
    fmatchers.append(fmatcher)
    obj_images.append(obj_image)

while True:
    isValid, img = video.read()
    if not isValid:
        break

    img_keypoints, img_descriptors = fdetector.detectAndCompute(img, None)

    max_index = -1
    max_inlier_num = 0
    tmp_match_count_arr = []
    for index, obj_pair in enumerate(obj_detector_pairs):
        obj_keypoints = obj_pair[0]
        obj_descriptors = obj_pair[1]

        match = fmatchers[index].match(img_descriptors, obj_descriptors)
        tmp_match_count_arr.append(len(match))
        if len(match) < min_inlier_num:
            continue

        if len(match) > max_inlier_num:
            max_inlier_num = len(match)
            max_keypoints = obj_keypoints
            max_descriptors = obj_descriptors
            max_match = match
            max_index = index

    # 매칭이 잘 된 이미지가 하나도 없다면 다시 시작
    if max_index == -1:
        continue

    obj_pts, img_pts = [], []
    for m in max_match:
        obj_pts.append(max_keypoints[m.trainIdx].pt)
        img_pts.append(img_keypoints[m.queryIdx].pt)
    obj_pts = np.array(obj_pts, dtype=np.float32)
    obj_pts = np.hstack((obj_pts, np.zeros((len(obj_pts), 1), dtype=np.float32))) # 2D -> 3D
    img_pts = np.array(img_pts, dtype=np.float32)

    if len(obj_pts) == 0 or len(img_pts) == 0:
        continue

    _, _, _, inliers = cv2.solvePnPRansac(obj_pts, img_pts, mtx, dist, useExtrinsicGuess=False,
                                                iterationsCount=200, reprojectionError=2., confidence=0.99)

    inlier_mask = np.zeros(len(max_match), dtype=np.uint8)
    inlier_mask[inliers] = 1

    img_result = cv2.drawMatches(img, img_keypoints, obj_images[max_index], max_keypoints, max_match, None, (0, 0, 255), (0, 127, 0), inlier_mask)
    
    cv2.putText(img_result, f"Match Count {tmp_match_count_arr[0]}, {tmp_match_count_arr[1]}, {tmp_match_count_arr[2]}, {tmp_match_count_arr[3]}", (10, 25), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))
    cv2.imshow('Image', img_result)

    key = cv2.waitKey(1)
    if key == ord(' '):
        try:
            cv2.imwrite(f'{REF_DIR}cur.png', img_result)

            obj_pts = obj_pts[:, :2]
            
            F, _ = cv2.findFundamentalMat(obj_pts, img_pts, cv2.FM_8POINT)
            K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
            E = K.T @ F @ K
            _, R, t, _ = cv2.recoverPose(E, obj_pts, img_pts)

            P0 = K @ np.eye(3, 4, dtype=np.float32)
            Rt = np.hstack((R, t))
            P1 = K @ Rt

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
                cur_rvec, _ = cv2.Rodrigues(R)
                cur_tvec = t
                reprojection_errors = calculate_reprojection_error(features_worldaxis, img_pts, cur_rvec, cur_tvec, mtx, dist)
                threshold = 0.1

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
    
    if key == ord('k'):
        break
    if key == ord('q'):
        video.release()
        cv2.destroyAllWindows()
        del data
        sys.exit()
del data