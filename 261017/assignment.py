# TODO: Morphological operation을 적용하여 최종 binary mask 생성
# TODO: Binary mask로 외곽선 검출
# TODO: 최종 농구공 영역 및 외곽선 선택
####################### assignment###########################

import cv2
from matplotlib import pyplot as plt
import numpy as np

lower_basketball_lab = np.array([70, 130, 70])
upper_basketball_lab = np.array([180, 230, 185])

basketball = cv2.imread('src/images/basketball_crop.jpg')

# LAB 색공간 변환
basketball_lab = cv2.cvtColor(basketball, cv2.COLOR_BGR2LAB)

l, a, b = cv2.split(basketball_lab)

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

l_clahe_enhanced = clahe.apply(l)
basketball_enhanced_lab = cv2.merge((l_clahe_enhanced, a, b))

mask_enhanced = cv2.inRange(
    basketball_enhanced_lab,
    lower_basketball_lab,
    upper_basketball_lab
)

basketball_enhanced_rgb = cv2.cvtColor(
    basketball_enhanced_lab, cv2.COLOR_LAB2RGB
)

final_result = cv2.bitwise_and(
    basketball_enhanced_rgb,
    basketball_enhanced_rgb,
    mask=mask_enhanced
)

# Morphological Operation
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

ball_mask_opened = cv2.morphologyEx(
    mask_enhanced, cv2.MORPH_OPEN, kernel, iterations=3
)

ball_mask_closed = cv2.morphologyEx(
    ball_mask_opened, cv2.MORPH_CLOSE, kernel, iterations=2
)

final_ball_mask = ball_mask_closed.copy()

# Contour 검출
contours, _ = cv2.findContours(
    final_ball_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# 최종 농구공 영역 및 외곽선 선택
final_img = basketball_enhanced_rgb.copy()

if contours:
    largest_contour = max(contours, key=cv2.contourArea)

    # 가장 큰 Contour만 최종 마스크에 남기기
    final_ball_mask = np.zeros_like(ball_mask_closed)
    cv2.drawContours(
        final_ball_mask,
        [largest_contour],
        -1,
        255,
        cv2.FILLED
    )

    # 농구공 외곽선 표시
    cv2.drawContours(
        final_img,
        [largest_contour],
        -1,
        (255, 0, 0),
        2
    )

# 결과 출력
plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)
plt.imshow(final_ball_mask, cmap="gray")
plt.title("Final binary mask")

plt.subplot(1, 2, 2)
plt.imshow(final_img)
plt.title("Largest basketball contour")

for ax in plt.gcf().axes:
    ax.axis("off")

plt.savefig("261017/result.png", dpi=150, bbox_inches="tight")
plt.show()