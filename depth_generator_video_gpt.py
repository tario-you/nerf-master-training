import cv2
import numpy as np
import os
from math import ceil


def empty_folder(folder_path):
    # Check if the folder path exists
    if not os.path.exists(folder_path):
        print("Folder does not exist.")
        return

    # Iterate over the files and subdirectories in the folder
    for root, dirs, files in os.walk(folder_path, topdown=False):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            # Delete the file
            os.remove(file_path)
            print(f"Deleted file: {file_path}")

        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            # Delete the subdirectory
            os.rmdir(dir_path)
            print(f"Deleted directory: {dir_path}")

    print("Folder emptied successfully.")


# Usage example
folder_path = '/path/to/folder'  # Replace with the actual folder path
empty_folder(folder_path)


def depth_estimation(current_frame, next_frame):
    # Convert frames to grayscale
    gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    gray_next = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    # Compute the disparity map
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(gray_current, gray_next)

    # Normalize the disparity map for visualization
    normalized_disparity = cv2.normalize(
        disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return normalized_disparity


def get_frame_count(mp4_file):
    video = cv2.VideoCapture(mp4_file)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()
    return frame_count


def main():
    # Path to the input video
    video_path = "input_video.mp4"

    # Path to save the output frames and depth images
    output_folder = "output_frames/"
    empty_folder(output_folder)

    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Check if the video file was opened successfully
    if not video.isOpened():
        print("Error opening video file")
        return

    frame_count = 0
    total_frames = ceil(get_frame_count(video_path)/2.24)

    while True:
        print(f'{frame_count}/{total_frames}')
        # Read the current frame from the video
        ret, current_frame = video.read()

        # If the frame was not read successfully, we've reached the end of the video
        if not ret:
            break

        # Read the next frame from the video
        ret, next_frame = video.read()

        # If the next frame was not read successfully, we've reached the end of the video
        if not ret:
            break

        # Save the current frame as an image

        frame_path = output_folder + f"r_{frame_count}.png"
        cv2.imwrite(frame_path, current_frame)

        # Perform depth estimation
        depth_image = depth_estimation(current_frame, next_frame)

        # Save the depth image
        depth_path = output_folder + f"r_{frame_count}_depth_0001.png"
        cv2.imwrite(depth_path, depth_image)

        frame_count += 1

    # Release the video file and close any open windows
    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
