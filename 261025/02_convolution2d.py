import time

import cv2
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt


IMAGE_PATH = "src/images/seagull.jpg"

TARGET_WIDTH = 1280
TARGET_HEIGHT = 720


dx_edge_filter = np.array(
    [
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ],
    dtype=np.float32
)

dy_edge_filter = np.array(
    [
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]
    ],
    dtype=np.float32
)

dx_bias = -0.2
dy_bias = -0.2


def relu(x):
    return np.maximum(0, x)


def convolution2d(image, kernel, bias=0.0, stride=1, padding=0):
    kernel_height, kernel_width = kernel.shape

    padded_image = np.pad(
        image,
        ((padding, padding), (padding, padding)),
        mode="constant",
        constant_values=0
    )

    output_height = (padded_image.shape[0] - kernel_height) // stride + 1
    output_width = (padded_image.shape[1] - kernel_width) // stride + 1

    output = np.zeros((output_height, output_width), dtype=np.float32)

    for output_y in range(output_height):
        for output_x in range(output_width):
            start_y = output_y * stride
            start_x = output_x * stride

            image_region = padded_image[
                start_y:start_y + kernel_height,
                start_x:start_x + kernel_width
            ]

            weighted_sum = np.sum(image_region * kernel)

            output[output_y, output_x] = weighted_sum + bias

    return output


def convolution2d_cp(image, kernel, bias=0.0, stride=1, padding=0):
    image = cp.asarray(image, dtype=cp.float32)
    kernel = cp.asarray(kernel, dtype=cp.float32)

    kernel_height, kernel_width = kernel.shape

    padded_image = cp.pad(
        image,
        ((padding, padding), (padding, padding)),
        mode="constant",
        constant_values=0
    )

    output_height = (padded_image.shape[0] - kernel_height) // stride + 1
    output_width = (padded_image.shape[1] - kernel_width) // stride + 1

    output = cp.zeros((output_height, output_width), dtype=cp.float32)

    for kernel_y in range(kernel_height):
        for kernel_x in range(kernel_width):
            image_region = padded_image[
                kernel_y:kernel_y + stride * (output_height - 1) + 1:stride,
                kernel_x:kernel_x + stride * (output_width - 1) + 1:stride
            ]

            output += image_region * kernel[kernel_y, kernel_x]

    output += bias

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

    cpu_start = time.perf_counter()

    output_dx_np = convolution2d(
        image=input_np,
        kernel=dx_edge_filter,
        bias=dx_bias,
        stride=1,
        padding=1
    )
    output_dy_np = convolution2d(
        image=input_np,
        kernel=dy_edge_filter,
        bias=dy_bias,
        stride=1,
        padding=1
    )

    cpu_end = time.perf_counter()
    cpu_time = cpu_end - cpu_start

    relu_dx_np = relu(output_dx_np)
    relu_dy_np = relu(output_dy_np)

    input_cp = cp.asarray(input_np, dtype=cp.float32)
    dx_edge_filter_cp = cp.asarray(dx_edge_filter, dtype=cp.float32)
    dy_edge_filter_cp = cp.asarray(dy_edge_filter, dtype=cp.float32)

    warmup_image = cp.zeros((64, 64), dtype=cp.float32)
    _ = convolution2d_cp(warmup_image, dx_edge_filter_cp, bias=dx_bias, stride=1, padding=1)
    _ = convolution2d_cp(warmup_image, dy_edge_filter_cp, bias=dy_bias, stride=1, padding=1)

    cp.cuda.Stream.null.synchronize()

    gpu_start = time.perf_counter()

    output_dx_cp = convolution2d_cp(
        image=input_cp,
        kernel=dx_edge_filter_cp,
        bias=dx_bias,
        stride=1,
        padding=1
    )
    output_dy_cp = convolution2d_cp(
        image=input_cp,
        kernel=dy_edge_filter_cp,
        bias=dy_bias,
        stride=1,
        padding=1
    )

    cp.cuda.Stream.null.synchronize()

    gpu_end = time.perf_counter()
    gpu_time = gpu_end - gpu_start

    relu_dx_cp_np = cp.asnumpy(cp.maximum(output_dx_cp, 0))
    relu_dy_cp_np = cp.asnumpy(cp.maximum(output_dy_cp, 0))

    compute_speedup = cpu_time / gpu_time

    dx_match = np.allclose(relu_dx_np, relu_dx_cp_np, atol=1e-5)
    dy_match = np.allclose(relu_dy_np, relu_dy_cp_np, atol=1e-5)

    dx_max_diff = np.max(np.abs(relu_dx_np - relu_dx_cp_np))
    dy_max_diff = np.max(np.abs(relu_dy_np - relu_dy_cp_np))

    print("========== NumPy CPU ==========")
    print(f"Processing time: {cpu_time:.4f} seconds")
    print()

    print("========== CuPy GPU ==========")
    print(f"Processing time: {gpu_time:.4f} seconds")
    print()

    print("========== Comparison ==========")
    print(f"Compute-only speedup: {compute_speedup:.2f}x")
    print(f"Vertical result match:   {dx_match} (max diff: {dx_max_diff:.2e})")
    print(f"Horizontal result match: {dy_match} (max diff: {dy_max_diff:.2e})")

    plt.figure(figsize=(18, 8))

    plt.subplot(2, 3, 1), plt.imshow(input_np, cmap="gray"), plt.title("Input Image")
    plt.subplot(2, 3, 2), plt.imshow(relu_dx_np, cmap="gray"), plt.title("Vertical Convolution (CPU)")
    plt.subplot(2, 3, 3), plt.imshow(relu_dx_cp_np, cmap="gray"), plt.title("Vertical Convolution (GPU)")
    plt.subplot(2, 3, 4), plt.imshow(input_np, cmap="gray"), plt.title("Input Image")
    plt.subplot(2, 3, 5), plt.imshow(relu_dy_np, cmap="gray"), plt.title("Horizontal Convolution (CPU)")
    plt.subplot(2, 3, 6), plt.imshow(relu_dy_cp_np, cmap="gray"), plt.title("Horizontal Convolution (GPU)")

    for ax in plt.gcf().axes:
        ax.axis("off")

    plt.show()


if __name__ == "__main__":
    main()
