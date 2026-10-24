# 에어 드로잉 (Air Drawing)

웹캠 앞에서 손가락으로 허공에 그림을 그리는 프로그램입니다.
MediaPipe로 손 랜드마크를 검출하고, 펴진 손가락 조합으로 모드(그리기 / 이동 / 색 변경 / 지우기)를 바꿉니다.

## 사용법

| 손 모양 | 모드 | 동작 |
|---|---|---|
| 검지만 펴기 | `DRAW` | 검지 끝을 따라 선 그리기 |
| 검지 + 중지 | `MOVE` | 펜 들기 (이동만, 그리지 않음) |
| 검지 + 중지 + 약지 | `COLOR` | 펜 색 변경 (RED → GREEN → BLUE → YELLOW → WHITE) |
| 주먹 (잠깐 유지) | `CLEAR` | 그림 전부 지우기 |
| 그 외 (손바닥 등) | `MOVE` | 이동만 |

종료: `q`

## 실행

```bash
source .venv/bin/activate
python CVproject/main.py
```

- 카메라: `/dev/video0` (USB 웹캠, GStreamer 사용)
- 모델: `src/models/MediaPipe/hand_landmarker.task`

---

## 1. 노트북(02_Computer-Vision.ipynb)에서 가져온 부분

아래 코드는 수업 노트북 코드를 거의 그대로 사용했습니다.

| 코드 | 출처 | 역할 |
|---|---|---|
| `pipeline` (GStreamer 문자열) | USB 카메라를 640x480 / 30fps로 읽고, 버퍼를 1장만 유지해 지연을 줄임 |
| `calculate_angle()` | 랜드마크 세 점으로 가운데 관절의 각도를 계산 |
| `angle_threshold`, `finger_angle_points` | 관절 각도가 160° 이상이면 "펴진 손가락"으로 판정 |
| `HandLandmarker` 생성 | MediaPipe 손 검출기 (손 1개만 검출하도록 `num_hands=1`) |
| `cv2.KalmanFilter(4, 2)` 설정 | 상태 `[x, y, vx, vy]`, 측정 `[x, y]`의 등속 모델로 좌표 떨림 보정 |
| `kalman.predict()` → `kalman.correct()` | 매 프레임 예측 후 측정값으로 보정 |

---

## 2. 직접 구현한 부분 (내 아이디어)

### 2-1. 손가락 상태를 "개수" 대신 "리스트"로: `finger_states()`

```python
def finger_states(hand):
    return [calculate_angle(hand[i1], hand[i2], hand[i3]) >= angle_threshold
            for i1, i2, i3 in finger_angle_points]
```

- 노트북 M은 펴진 손가락의 **개수**만 셌습니다. 하지만 개수로는 "검지 1개"와 "소지 1개"를 구분할 수 없습니다.
- 그래서 판정 결과를 엄지부터 소지까지 순서대로 담은 **bool 리스트**로 반환하도록 바꿨습니다.
  - 예: 검지만 폈으면 `[False, True, False, False, False]`
- 판정 방식(관절 각도 ≥ 160°) 자체는 노트북과 같습니다.

### 2-2. 손 모양 → 모드 변환: `get_mode()`

```python
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
    return "MOVE"
```

- **엄지는 무시**합니다. 엄지는 다른 손가락과 굽히는 방향이 달라서 각도 판정이 자주 틀렸기 때문입니다. 검지~소지 4개만으로 모드를 구분합니다.
- 손가락을 하나씩 더 펼 때마다 다음 모드가 되도록 정해서 외우기 쉽게 했습니다 (1개 = 그리기, 2개 = 이동, 3개 = 색).
- 어디에도 해당하지 않는 모양(손바닥 펴기 등)은 **MOVE**로 처리합니다. 애매한 손 모양에서는 그림이 그려지지 않게 하는 안전한 기본값입니다.

### 2-3. 손 모양 오작동 방지: 프레임 유지 카운터

```python
same_frames = same_frames + 1 if mode == last_mode else 0

if mode == "COLOR" and same_frames == COLOR_HOLD:   # 5 프레임
    color_idx = (color_idx + 1) % len(PEN_COLORS)
if mode == "CLEAR" and same_frames == CLEAR_HOLD:   # 15 프레임
    canvas[:] = 0
```

- 손을 움직이는 도중에 한두 프레임 동안 다른 모양으로 잘못 인식되는 경우가 있습니다. 이때 바로 색이 바뀌거나 그림이 지워지면 안 됩니다.
- `same_frames`로 **같은 모드가 몇 프레임 연속 유지됐는지**를 셉니다. 모드가 바뀌면 0으로 초기화합니다.
- 조건을 `>=`가 아닌 `==`로 걸었습니다. 그래서 손 모양을 계속 유지해도 **딱 한 번만** 실행됩니다. `>=`였다면 COLOR 모양을 유지하는 동안 매 프레임 색이 바뀌었을 것입니다.
- CLEAR는 되돌릴 수 없는 동작이라 COLOR(5)보다 훨씬 긴 15 프레임(약 0.5초)을 요구합니다.

### 2-4. 선 그리기: `canvas` + `prev_point`

```python
if mode == "DRAW":
    if prev_point is not None:
        cv2.line(canvas, prev_point, point, PEN_COLORS[color_idx][1], PEN_THICKNESS)
    prev_point = point
else:
    prev_point = None
```

- 카메라 프레임은 매번 새로 들어오므로, 그림은 별도의 검은 도화지 `canvas`에 누적해서 그립니다. `canvas`는 첫 프레임과 같은 크기로 생성합니다.
- 점을 찍지 않고 **직전 위치(`prev_point`)에서 현재 위치까지 선**을 긋습니다. 손이 빠르게 움직여도 그림이 점선으로 끊기지 않습니다.
- DRAW가 아닐 때는 `prev_point = None`으로 선을 끊습니다. 다시 DRAW가 됐을 때 이전 위치에서 엉뚱한 선이 이어지지 않게 하기 위해서입니다.

### 2-5. 그림을 카메라 화면 위에 합치기: `overlay()`

```python
def overlay(frame, canvas):
    drawn = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY) > 0
    result = frame.copy()
    result[drawn] = canvas[drawn]
    return result
```

- `canvas`를 회색조로 바꿔서 0보다 큰 픽셀, 즉 **실제로 그려진 곳**만 True인 마스크를 만듭니다.
- 그 위치의 픽셀만 카메라 화면에 덮어씁니다. 그래서 그림은 선명하게 보이고 나머지는 카메라 화면 그대로입니다.
- 이 방식 때문에 펜 색에 **검정 `(0,0,0)`은 넣지 않았습니다**. 검정으로 그리면 "안 그린 곳"과 구분할 수 없습니다.

### 2-6. Kalman 필터 적용 방식

노트북의 Kalman 설정을 손가락 끝(랜드마크 8번) 추적에 맞게 적용했습니다.

```python
if not found:
    init = np.array([[tip[0]], [tip[1]], [0.0], [0.0]], dtype=np.float32)
    kalman.statePre, kalman.statePost = init, init.copy()
    found = True
```

- **첫 검출 시 초기화:** 손을 처음 찾으면 상태를 "현재 위치, 속도 0"으로 맞춥니다. 노트북은 `statePre`만 설정했지만, 여기서는 `statePost`도 함께 설정합니다. 첫 `predict()`가 `statePost`에서 출발하기 때문에, 이렇게 해야 (0,0)에서 선이 날아오지 않습니다.
- **손이 사라지면 리셋:** 노트북은 물체가 가려지면 `predict()`만으로 위치를 계속 추정했습니다. 그림 그리기에서는 손이 안 보이는 동안 위치를 추정하면 다시 나타났을 때 엉뚱한 곳에서 선이 시작됩니다. 그래서 손이 사라지면 `found = False`, `prev_point = None`으로 **선을 끊고 Kalman을 처음부터 다시 시작**하게 했습니다.

### 2-7. 화면 처리와 표시

- `cv2.flip(frame, 1)`: 좌우 반전합니다. 손을 오른쪽으로 움직이면 화면에서도 오른쪽으로 가서 거울처럼 자연스럽습니다.
- `cv2.resize(..., fx=SCALE, fy=SCALE)`: 화면을 절반 크기로 줄여 MediaPipe 처리 속도를 높입니다.
- 펜 위치 표시: DRAW 모드에서는 꽉 찬 원, 다른 모드에서는 빈 원으로 그려서 지금 그리는 중인지 바로 알 수 있습니다.
- 왼쪽 위에 `현재 색 | 모드`를 현재 펜 색으로 표시합니다.

## 설정값 (main.py 상단)

| 변수 | 기본값 | 설명 |
|---|---|---|
| `SCALE` | 0.5 | 화면 크기 배율 |
| `PEN_THICKNESS` | 4 | 선 두께 |
| `COLOR_HOLD` | 5 | 색 변경에 필요한 유지 프레임 수 |
| `CLEAR_HOLD` | 15 | 지우기에 필요한 유지 프레임 수 |
| `angle_threshold` | 160 | 손가락 펴짐 판정 각도 |
| `measurementNoiseCov` 배수 | 1.0 | 클수록 부드럽지만 반응이 느림 |
