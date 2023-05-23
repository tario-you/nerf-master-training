import cv2
import numpy as np


def depth_estimation(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute the disparity map
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(gray, gray)

    # Normalize the disparity map for visualization
    normalized_disparity = cv2.normalize(
        disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return normalized_disparity


def main():
    # Path to the input image
    image_path = "input_img.png"

    # Path to save the output depth image
    output_path = "depth_img.png"

    # Read the input image
    image = cv2.imread(image_path)

    # Check if the image was read successfully
    if image is None:
        print("Error opening image file")
        return

    # Perform depth estimation
    depth_image = depth_estimation(image)

    # Save the depth image
    cv2.imwrite(output_path, depth_image)

    print("Depth image saved successfully.")


if __name__ == "__main__":
    main()
