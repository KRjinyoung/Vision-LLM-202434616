\# 1017 과제 - 농구공 영역 검출



\#### 실수로 커밋 안해서 제출이 늦었습니다.



\## 실행 환경

\- Python 3.10

\- Ubuntu (WSL)



\## 필요한 라이브러리

\- numpy==1.23.5

\- opencv-python==4.7.0.72

\- matplotlib==3.7.3



\## 설치 방법

```bash

pip install numpy==1.23.5 opencv-python==4.7.0.72 matplotlib==3.7.3

```



\## 실행 방법

```bash

python 1017.py

```



\## 참고 사항

\- 입력 이미지: `src/images/basketball\_crop.jpg`

\- LAB 색공간 변환 및 CLAHE 적용

\- Opening 3회, Closing 2회로 노이즈 제거

\- 가장 큰 Contour를 농구공 영역으로 선택

\- 실행 결과는 `result.png`로 저장



