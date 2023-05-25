import cv2
import os


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


def extract_frames(video_path, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Read the first frame
    success, frame = video.read()
    frame_count = 0

    while success:
        # Save the current frame as an image
        output_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
        cv2.imwrite(output_path, frame)

        # Read the next frame
        success, frame = video.read()
        frame_count += 1

    # Release the video file
    video.release()

    print(f"Frames extracted: {frame_count}")


# Specify the path to the video file
video_path = "/Users/tarioyou/code/nerf-master-training/input_video_short.mp4"

# Specify the output folder
output_folder = "/Users/tarioyou/code/nerf-master-training/input_original"

empty_folder(output_folder)

# Call the function to extract frames
extract_frames(video_path, output_folder)
