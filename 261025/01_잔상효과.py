import time

import cv2
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt


IMAGE_PATH = "src/images/seagull.jpg"

TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

TRAIL_LENGTH = 32


def motion_trail_np(image, weights):
    height, width = image.shape[:2]
    output = np.zeros_like(image, dtype=np.float32)
    trail_length = len(weights)

    for output_y in range(height):
        for output_x in range(width):
            accumulated = 0.0

            for shift in range(trail_length):
                input_x = output_x - shift

                if input_x < 0:
                    break

                accumulated += image[output_y, input_x] * weights[shift]

            output[output_y, output_x] = accumulated

    return output


def motion_trail_cp(image, weights):
    image = cp.asarray(image, dtype=cp.float32)
    weights = cp.asarray(weights, dtype=cp.float32)

    height, width = image.shape[:2]
    output = cp.zeros_like(image, dtype=cp.float32)
    trail_length = len(weights)

    for shift in range(trail_length):
        if shift >= width:
            break

        input_region = image[:, :width - shift]
        output_region = output[:, shift:]

        output_region += input_region * weights[shift]

    return output


def main():
    img_gray = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

    if img_gray is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {IMAGE_PATH}")

    img_gray = cv2.resize(
        img_gray,
        (TARGET_WIDTH, TARGET_HEIGHT),
        interpolation=cv2.INTER_AREA
    )

    input_np = img_gray.astype(np.float32) / 255.0

    weights_np = np.linspace(
        1.0,
        0.05,
        TRAIL_LENGTH,
        dtype=np.float32
    )
    weights_np /= np.sum(weights_np)

    cpu_start = time.perf_counter()

    output_np = motion_trail_np(
        image=input_np,
        weights=weights_np
    )

    cpu_end = time.perf_counter()
    cpu_time = cpu_end - cpu_start

    upload_start = time.perf_counter()

    input_cp = cp.asarray(input_np)
    weights_cp = cp.asarray(weights_np)

    cp.cuda.Stream.null.synchronize()

    upload_end = time.perf_counter()
    upload_time = upload_end - upload_start

    warmup_image = cp.zeros((64, 64), dtype=cp.float32)
    warmup_weights = cp.asarray(weights_np[:4])

    _ = motion_trail_cp(
        image=warmup_image,
        weights=warmup_weights
    )

    cp.cuda.Stream.null.synchronize()

    gpu_start = time.perf_counter()

    output_cp = motion_trail_cp(
        image=input_cp,
        weights=weights_cp
    )

    cp.cuda.Stream.null.synchronize()

    gpu_end = time.perf_counter()
    gpu_time = gpu_end - gpu_start

    download_start = time.perf_counter()

    output_cp_np = cp.asnumpy(output_cp)

    download_end = time.perf_counter()
    download_time = download_end - download_start

    compute_speedup = cpu_time / gpu_time

    gpu_total_time = upload_time + gpu_time + download_time
    total_speedup = cpu_time / gpu_total_time

    print("========== NumPy CPU ==========")
    print(f"Processing time: {cpu_time:.4f} seconds")
    print()

    print("========== CuPy GPU ==========")
    print(f"CPU to GPU: {upload_time:.4f} seconds")
    print(f"GPU processing: {gpu_time:.4f} seconds")
    print(f"GPU to CPU: {download_time:.4f} seconds")
    print(f"Total GPU time: {gpu_total_time:.4f} seconds")
    print()

    print("========== Comparison ==========")
    print(f"Compute-only speedup: {compute_speedup:.2f}x")
    print(f"Including transfers: {total_speedup:.2f}x")

    plt.figure(figsize=(18, 12))

    plt.subplot(1, 3, 1), plt.imshow(input_np, cmap="gray"), plt.title("Original Image")
    plt.subplot(1, 3, 2), plt.imshow(output_np, cmap="gray"), plt.title("Motion Trail (CPU)")
    plt.subplot(1, 3, 3), plt.imshow(output_cp_np, cmap="gray"), plt.title("Motion Trail (GPU)")

    for ax in plt.gcf().axes:
        ax.axis("off")

    plt.figure(figsize=(12, 8))
    plt.imshow(output_cp_np, cmap="gray")
    plt.axis("off")

    plt.show()


if __name__ == "__main__":
    main()
