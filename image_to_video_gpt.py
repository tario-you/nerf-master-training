import cv2
import os
from tqdm import tqdm


def sort_images(image_files):
    def extract_number(filename):
        return int(filename.split('_')[1].split('.png')[0])

    return sorted(image_files, key=extract_number)


def images_to_video(input_folder, output_file, fps):
    # Get the list of image file names in the input folder
    image_files = [f for f in os.listdir(
        input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    sorted_image_files = sort_images(image_files)

    print(sorted_image_files)

    # Determine the width and height of the images by reading the first image
    first_image = cv2.imread(os.path.join(input_folder, sorted_image_files[0]))
    height, width, _ = first_image.shape

    # Define the video codec and create a VideoWriter object
    # You can change the codec as needed
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # Calculate the duration of each frame based on the frame rate
    frame_duration = 1 / fps

    # Iterate through each image and write it as a frame in the video
    for i in tqdm(range(len(sorted_image_files))):
        image_file = sorted_image_files[i]
        image_path = os.path.join(input_folder, image_file)
        frame = cv2.imread(image_path)
        video_writer.write(frame)
        # Wait for the calculated duration
        cv2.waitKey(int(frame_duration * 1000))

    # Release the VideoWriter object and close any open windows
    video_writer.release()
    cv2.destroyAllWindows()


# Specify the input folder containing the images
input_folder = '/Users/tarioyou/Downloads/nerf-master/drum_test_no_depth'

# Specify the output video file name
output_file = 'output_video.mp4'

# Specify the desired frames per second (fps) for the output video
fps = 30

# Convert the images to a video
images_to_video(input_folder, output_file, fps)
