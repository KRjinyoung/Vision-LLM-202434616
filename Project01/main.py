"""
Section 2 Project: 에어 드로잉 (허공에 손가락으로 그림 그리기)

손 모양으로 모드 선택 (엄지는 판정이 불안정해서 제외, 검지~소지 4개만 사용)
- 검지만 펴기         : DRAW  (검지 끝을 따라 선 그리기)
- 검지 + 중지         : MOVE  (펜 들기, 이동만)
- 검지 + 중지 + 약지  : COLOR (펜 색 바꾸기)
- 주먹 (잠깐 유지)    : CLEAR (전부 지우기)

종료: q
"""
import os

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ------------------------- 설정 -------------------------
SCALE = 0.5         # 화면 크기 배율 (0.5 = 가로세로 절반)

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "models", "MediaPipe", "hand_landmarker.task")

# 펜 색 (BGR). 검정(0,0,0)은 "안 그린 곳"과 구분이 안 되므로 쓰지 않음
PEN_COLORS = [("RED", (0, 0, 255)), ("GREEN", (0, 255, 0)), ("BLUE", (255, 0, 0)),
              ("YELLOW", (0, 255, 255)), ("WHITE", (255, 255, 255))]
PEN_THICKNESS = 4

COLOR_HOLD = 5      # COLOR 손모양을 몇 프레임 유지해야 색이 바뀌는지
CLEAR_HOLD = 15     # 주먹을 몇 프레임 유지해야 지워지는지 (실수로 지우는 것 방지)


# ---------------- 손가락 각도 계산 ----------------
# 중간 점 기준으로 각도 계산하는 함수 (mp02.py 그대로)
def calculate_angle(p1, p2, p3):
    vector1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
    vector2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])

    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    cosine = np.dot(vector1, vector2) / (magnitude1 * magnitude2)
    cosine = np.clip(cosine, -1.0, 1.0)

    return np.degrees(np.arccos(cosine))


angle_threshold = 160  # 손가락 펼쳐짐 여부 판단 각도 임계값

# 각 손가락의 각도를 계산할 landmark index
finger_angle_points = (
    (1, 2, 3),      # 엄지: 2번 중심
    (5, 6, 7),      # 검지: 6번 중심
    (9, 10, 11),    # 중지: 10번 중심
    (13, 14, 15),   # 약지: 14번 중심
    (17, 18, 19),   # 소지: 18번 중심
)


# 규칙 기반 손가락 판정: 개수 대신 "어떤 손가락이 펴졌는지"를 리스트로 반환
# 예: 검지만 폈으면 [False, True, False, False, False]
def finger_states(hand):
    return [calculate_angle(hand[i1], hand[i2], hand[i3]) >= angle_threshold
            for i1, i2, i3 in finger_angle_points]


# 펴진 손가락 조합 -> 모드 (규칙 기반)
def get_mode(states):
    _, index, middle, ring, pinky = states  # 엄지는 무시

    if index and not middle and not ring and not pinky:
        return "DRAW"
    if index and middle and not ring and not pinky:
        return "MOVE"
    if index and middle and ring and not pinky:
        return "COLOR"
    if not index and not middle and not ring and not pinky:
        return "CLEAR"
    return "MOVE"  # 손바닥 펴기 등 나머지는 이동만


# 그림판(canvas)에서 그려진 픽셀만 카메라 화면 위에 덮어쓰기
def overlay(frame, canvas):
    drawn = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY) > 0  # 그려진 곳 = True
    result = frame.copy()
    result[drawn] = canvas[drawn]
    return result


# ---------------- USB 카메라 GStreamer 파이프라인 ----------------
pipeline = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 "
    "max-size-bytes=0 max-size-time=0 ! "
    "appsink drop=true max-buffers=1 sync=false"
)


# ---------------- OpenCV Kalman Filter ----------------
# 손가락 끝의 떨림을 줄이기 위해 사용
# 상태 벡터: [x, y, vx, vy] / 측정 벡터: [x, y]
kalman = cv2.KalmanFilter(4, 2)
kalman.transitionMatrix = np.array(
    [[1, 0, 1, 0],
     [0, 1, 0, 1],
     [0, 0, 1, 0],
     [0, 0, 0, 1]], dtype=np.float32)
kalman.measurementMatrix = np.array(
    [[1, 0, 0, 0],
     [0, 1, 0, 0]], dtype=np.float32)
kalman.processNoiseCov = np.diag([1e-2, 1e-2, 5e-2, 5e-2]).astype(np.float32)
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0  # 클수록 더 부드럽지만 반응이 느려짐
kalman.errorCovPost = np.eye(4, dtype=np.float32)


# ---------------- MediaPipe 모델 준비 (노트북 M) ----------------
base_option = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(base_options=base_option, num_hands=1)
hand_detector = vision.HandLandmarker.create_from_options(options)


# ---------------- 메인 루프 ----------------
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

canvas = None            # 그림이 그려지는 검은 도화지 (첫 프레임 크기로 생성)
color_idx = 0            # 현재 펜 색 번호
prev_point = None        # 직전 프레임의 펜 위치 (선을 이을 시작점)
found = False            # 손을 찾아서 Kalman이 시작됐는지
last_mode, same_frames = None, 0

while True:
    ret, frame = cap.read()

    if not ret:
        break
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # 1) 이미지 조정: 거울처럼 좌우 반전 + 크기 축소 + RGB 변환 (MediaPipe 입력용)
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, None, fx=SCALE, fy=SCALE)
    h, w = frame.shape[:2]
    if canvas is None:
        canvas = np.zeros_like(frame)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 2) 손 검출
    result = hand_detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

    if len(result.hand_landmarks) == 0:
        # 손이 사라지면 선을 끊고 Kalman도 처음부터 다시 시작
        mode, prev_point, found = "NO HAND", None, False
    else:
        hand = result.hand_landmarks[0]
        mode = get_mode(finger_states(hand))

        # 3) 검지 끝(8번) 좌표 -> 픽셀 좌표
        tip = (int(hand[8].x * w), int(hand[8].y * h))

        # 4) Kalman으로 손가락 끝 떨림 보정
        if not found:
            # 처음 찾았을 때 현재 위치로 상태 초기화 (속도는 0)
            init = np.array([[tip[0]], [tip[1]], [0.0], [0.0]], dtype=np.float32)
            kalman.statePre, kalman.statePost = init, init.copy()
            found = True
        kalman.predict()                                                                  # 예측
        state = kalman.correct(np.array([[tip[0]], [tip[1]]], dtype=np.float32))          # 보정
        point = (int(state[0, 0]), int(state[1, 0]))

        # 5) 같은 모드가 몇 프레임 유지됐는지 (COLOR, CLEAR 오작동 방지)
        same_frames = same_frames + 1 if mode == last_mode else 0

        if mode == "DRAW":
            if prev_point is not None:
                cv2.line(canvas, prev_point, point, PEN_COLORS[color_idx][1], PEN_THICKNESS)
            prev_point = point
        else:
            prev_point = None  # 그리기가 아니면 선을 끊음

        if mode == "COLOR" and same_frames == COLOR_HOLD:   # 딱 한 번만 바뀌도록 ==
            color_idx = (color_idx + 1) % len(PEN_COLORS)
        if mode == "CLEAR" and same_frames == CLEAR_HOLD:
            canvas[:] = 0

        # 펜 위치 표시: 그리는 중이면 꽉 찬 원, 아니면 빈 원
        cv2.circle(frame, point, 6, PEN_COLORS[color_idx][1], -1 if mode == "DRAW" else 2)

    last_mode = mode

    # 6) 그림 + 카메라 화면 합치기, 상태 표시
    output = overlay(frame, canvas)
    cv2.putText(output, f"{PEN_COLORS[color_idx][0]} | {mode}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, PEN_COLORS[color_idx][1], 1)
    cv2.imshow("Air Drawing", output)

hand_detector.close()
cap.release()
cv2.destroyAllWindows()
