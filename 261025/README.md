# 261025 과제 TODO

colab 환경에서 실험하였습니다
GPU = tesla T4
라이브러리가 꼬여 실행에 오류가있으면 ipynb참고 부탁드립니다

## 1. 잔상효과 — `motion_trail_np` / `motion_trail_cp`

**역할**
- `motion_trail_np`: CPU(NumPy)로 잔상 효과 생성. 픽셀 단위 for문.
- `motion_trail_cp`: GPU(CuPy)로 같은 잔상 효과를 배열 단위 연산으로 생성.

출력 픽셀 = 현재 + 이전 32개 픽셀의 가중합 (가중치는 `linspace(1.0, 0.05)`를 합 1로 정규화).

- **NumPy**: 픽셀마다 `y, x, shift` 3중 for문. `input_x < 0`이면 break.
- **CuPy**: 픽셀 루프 없이 `shift`만 반복. 이미지를 밀어서 통째로 누적.

```python
output[:, shift:] += image[:, :width - shift] * weights[shift]
```

**결과**
========== NumPy CPU ==========
Processing time: 17.3546 seconds

========== CuPy GPU ==========
CPU to GPU: 0.0036 seconds
GPU processing: 0.0046 seconds
GPU to CPU: 0.0015 seconds
Total GPU time: 0.0097 seconds

========== Comparison ==========
Compute-only speedup: 3757.41x
Including transfers: 1793.46x

## 2. Convolution2D — `convolution2d` / `convolution2d_cp`

**역할**
- `convolution2d`: CPU(NumPy)로 2D 합성곱 계산. 출력 위치마다 커널을 곱해 합산.
- `convolution2d_cp`: GPU(CuPy)로 같은 합성곱을 커널 원소 단위 벡터 연산으로 계산.

`stride`, `padding`, `bias` 지원. 출력 크기는 `(H + 2P - K) // S + 1`.

- **NumPy**: 출력 위치마다 영역을 잘라 `np.sum(region * kernel)`.
- **CuPy**: 출력 위치 대신 **커널 원소(3×3 = 9번)** 만 순회. stride 슬라이싱(`a:b:stride`)으로 전체 출력을 한 번에 계산.

**결과**
========== NumPy CPU ==========
Processing time: 10.9710 seconds

========== CuPy GPU ==========
Processing time: 0.0191 seconds

========== Comparison ==========
Compute-only speedup: 574.69x
Vertical result match:   True (max diff: 1.19e-07)
Horizontal result match: True (max diff: 2.38e-07)


## 3. main_gpu — `main_gpu()`

**역할**
- `main_gpu`: MNIST CNN 학습 전체(데이터 로드, 학습, 평가)를 GPU에서 실행하고 시간을 측정. CPU 버전 `main_cpu`와 속도 비교용.

`main_cpu()`와 같은 학습 코드를 `cuda` 디바이스로 옮겨 학습 시간을 비교.

- `torch.cuda.manual_seed_all`로 seed를 맞춰 같은 조건에서 비교.
- GPU는 비동기 실행이라 시간 측정 전후에 `torch.cuda.synchronize()` 호출.
- 배치마다 `images.to(device)`, `labels.to(device)`.


**결과**
=== CPU Training ===
Epoch 1/3 | Train loss: 0.3133 | Train accuracy: 90.52% | Test loss: 0.0953 | Test accuracy: 96.99% | Time: 18.17s
Epoch 2/3 | Train loss: 0.0925 | Train accuracy: 97.15% | Test loss: 0.0693 | Test accuracy: 97.73% | Time: 19.17s
Epoch 3/3 | Train loss: 0.0670 | Train accuracy: 97.89% | Test loss: 0.0508 | Test accuracy: 98.20% | Time: 22.90s
Total CPU training time: 60.23s

=== GPU Training ===
Epoch 1/3 | Train loss: 0.3132 | Train accuracy: 90.54% | Test loss: 0.0963 | Test accuracy: 96.93% | Time: 9.36s
Epoch 2/3 | Train loss: 0.0924 | Train accuracy: 97.11% | Test loss: 0.0695 | Test accuracy: 97.72% | Time: 12.40s
Epoch 3/3 | Train loss: 0.0669 | Train accuracy: 97.89% | Test loss: 0.0492 | Test accuracy: 98.22% | Time: 10.69s
Total GPU training time: 32.45s

=== CPU vs. GPU Comparison ===
CPU training time: 60.23s
GPU training time: 32.45s
Time difference: 27.78s
GPU speedup: 1.86x


## 4. CIFAR-10 — `ConvBlock`, `CIFAR10CNN`

**역할**
- `ConvBlock`: 특징 추출 기본 단위. 채널 수를 늘리고 `MaxPool`로 해상도를 줄임.
- `CIFAR10CNN`: `ConvBlock`을 쌓아 이미지를 10개 클래스로 분류하는 모델.

- `ConvBlock`: `Conv - BN - ReLU` ×2 + `MaxPool`. 3단(64→128→256)으로 쌓음.
- GAP + `Dropout(0.3)` + `Linear(256, 10)`.
- 학습 증강(`RandomCrop`, `RandomHorizontalFlip`) + `Normalize`, `AdamW(lr=1e-3)`로 5 epoch 학습 후 `.pth` 저장.


## 5. IoU — `IoU.py`

**TODO 1)** `convert_bbox_coord`: `(x, y, w, h)` → `(x1, y1, x2, y2)`
**역할**: contour로 얻은 `(x, y, w, h)` 박스를 IoU 계산에 쓰는 좌상단/우하단 좌표 형식으로 변환.

```python
return (x, y, x + w, y + h)
```

**TODO 2)** `calculate_iou`: 교집합 / 합집합
**역할**: 두 박스가 얼마나 겹치는지 0~1 값으로 계산. NMS에서 중복 박스 판단 기준.

```python
w = max(0, min(xa_2, xb_2) - max(xa_1, xb_1))   # 안 겹치면 0
h = max(0, min(ya_2, yb_2) - max(ya_1, yb_1))
iou = w * h / (area_a + area_b - w * h)
```


## 6. NMS — `MNS.py`

**TODO) Confidence Thresholding** `prune_low_conf`
**역할**: 신뢰도가 낮은 박스를 NMS 전에 걸러내 후보를 줄임. score가 `conf_threshold`보다 낮은 박스를 제거 (`score >= threshold`만 유지).

**TODO) NMS** `apply_nms`
**역할**: 같은 물체를 가리키는 겹친 박스 중 score가 가장 높은 하나만 남김.
1. `np.argsort(scores)[::-1]`로 score 내림차순 인덱스 생성
2. 가장 높은 박스를 결과에 추가
3. 그 박스와 IoU가 `iou_threshold` 이상인 박스를 후보에서 제거
4. 후보가 없어질 때까지 반복
