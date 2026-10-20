import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Segment a basketball, clean its binary mask, and select its contour."
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=project_dir / "src" / "images" / "basketball_crop.jpg",
        help="Path to the input image.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "src" / "output" / "basketball_contour",
        help="Directory where result images are saved.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    image_bgr = cv2.imread(str(args.image))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read input image: {args.image}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Apply CLAHE to lightness, then segment with the notebook's LAB range.
    image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(image_lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_lab = cv2.merge((clahe.apply(lightness), channel_a, channel_b))
    lower_basketball_lab = np.array([70, 130, 70], dtype=np.uint8)
    upper_basketball_lab = np.array([180, 230, 185], dtype=np.uint8)
    initial_mask = cv2.inRange(
        enhanced_lab, lower_basketball_lab, upper_basketball_lab
    )

    # Opening removes small noise; closing fills gaps inside the ball region.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened_mask = cv2.morphologyEx(
        initial_mask, cv2.MORPH_OPEN, kernel, iterations=1
    )
    final_mask = cv2.morphologyEx(
        opened_mask, cv2.MORPH_CLOSE, kernel, iterations=2
    )

    contours, _ = cv2.findContours(
        final_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise RuntimeError(
            "No basketball contour found. Adjust the LAB thresholds for this image."
        )

    basketball_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(basketball_contour) == 0:
        raise RuntimeError("The selected contour has no area.")

    basketball_mask = np.zeros_like(final_mask)
    cv2.drawContours(
        basketball_mask, [basketball_contour], contourIdx=-1, color=255, thickness=cv2.FILLED
    )
    basketball_region = cv2.bitwise_and(image_rgb, image_rgb, mask=basketball_mask)

    outline_image = image_rgb.copy()
    cv2.drawContours(
        outline_image, [basketball_contour], contourIdx=-1, color=(0, 255, 0), thickness=3
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output_dir / "final_mask.png"), basketball_mask)
    cv2.imwrite(
        str(args.output_dir / "basketball_outline.png"),
        cv2.cvtColor(outline_image, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        str(args.output_dir / "basketball_region.png"),
        cv2.cvtColor(basketball_region, cv2.COLOR_RGB2BGR),
    )

    x, y, width, height = cv2.boundingRect(basketball_contour)
    print(f"Selected contour area: {cv2.contourArea(basketball_contour):.0f} px")
    print(f"Bounding box: x={x}, y={y}, width={width}, height={height}")
    print(f"Saved results to: {args.output_dir}")

    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title("Original")
    axes[0, 1].imshow(initial_mask, cmap="gray")
    axes[0, 1].set_title("Initial LAB mask")
    axes[1, 0].imshow(basketball_mask, cmap="gray")
    axes[1, 0].set_title("Final basketball mask")
    axes[1, 1].imshow(outline_image)
    axes[1, 1].set_title("Selected contour")
    for axis in axes.flat:
        axis.axis("off")
    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()