import numpy as np
import cv2
import os
import statistics


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


def get_frame_count(mp4_file):
    video = cv2.VideoCapture(mp4_file)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()
    return frame_count


if __name__ == "__main__":
    # Reading the mapping values for stereo image rectification
    cv_file = cv2.FileStorage(
        "data/stereo_rectify_maps.xml", cv2.FILE_STORAGE_READ)
    Left_Stereo_Map_x = cv_file.getNode("Left_Stereo_Map_x").mat()
    Left_Stereo_Map_y = cv_file.getNode("Left_Stereo_Map_y").mat()
    Right_Stereo_Map_x = cv_file.getNode("Right_Stereo_Map_x").mat()
    Right_Stereo_Map_y = cv_file.getNode("Right_Stereo_Map_y").mat()
    cv_file.release()

    cv2.namedWindow('disp', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('disp', 600, 600)

    preFilterType = 1  # 1,1

    numDisparities = (range(1, 17))
    blockSize = (range(5, 50))
    preFilterSize = (range(2, 25))
    preFilterCap = (range(5, 62))
    textureThreshold = (range(10, 100))
    uniquenessRatio = (range(15, 100))
    speckleRange = (range(0, 100))
    speckleWindowSize = (range(3, 25))
    disp12MaxDiff = (range(5, 25))
    minDisparity = (range(1, 25))

    values = [numDisparities, blockSize, preFilterSize, preFilterCap,
              textureThreshold, uniquenessRatio, speckleRange, speckleWindowSize, disp12MaxDiff, minDisparity]

    stereo = cv2.StereoBM_create()

    video_path = "input_video.mp4"
    output_folder = "/Users/tarioyou/code/nerf-master-training/output_frames2"
    empty_folder(output_folder)
    video = cv2.VideoCapture(video_path)

    # for general parameter tuning

    imgR = cv2.imread(
        '/Users/tarioyou/code/nerf-master-training/drum_test_no_depth/r_0.png')
    imgL = cv2.imread(
        '/Users/tarioyou/code/nerf-master-training/drum_test_no_depth/r_1.png')

    imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)

    for i in range(len(values)+1):
        var = [min(v) for v in values]
        if i != len(values):
            var[i] = max(values[i])
        numDisparities = var[0]
        blockSize = var[1]
        preFilterSize = var[2]
        preFilterCap = var[3]
        textureThreshold = var[4]
        uniquenessRatio = var[5]
        speckleRange = var[6]
        speckleWindowSize = var[7]
        disp12MaxDiff = var[8]
        minDisparity = var[9]

        print(var)

        # Applying stereo image rectification on the left image
        Left_nice = cv2.remap(imgL_gray,
                              Left_Stereo_Map_x,
                              Left_Stereo_Map_y,
                              cv2.INTER_LANCZOS4,
                              cv2.BORDER_CONSTANT,
                              0)

        # Applying stereo image rectification on the right image
        Right_nice = cv2.remap(imgR_gray,
                               Right_Stereo_Map_x,
                               Right_Stereo_Map_y,
                               cv2.INTER_LANCZOS4,
                               cv2.BORDER_CONSTANT,
                               0)

        disparity = stereo.compute(Left_nice, Right_nice)

        disparity = disparity.astype(np.float32)

        disparity = (disparity/16.0 - minDisparity)/numDisparities

        var = [str(v) for v in var]

        output_image_path = output_folder+f"/disp_{'_'.join(var)}.png"

        cv2.imwrite(output_image_path, disparity)

    while False:  # for video
        frame_count += 1
        print(f'{frame_count}/{total_frames}')
        if frame_count != 1:
            current_frame = next_frame
        ret, next_frame = video.read()
        if not ret:
            break

        imgR = current_frame
        imgL = next_frame

        # Proceed only if the frames have been captured
        imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
        imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)

        # Applying stereo image rectification on the left image
        Left_nice = cv2.remap(imgL_gray,
                              Left_Stereo_Map_x,
                              Left_Stereo_Map_y,
                              cv2.INTER_LANCZOS4,
                              cv2.BORDER_CONSTANT,
                              0)

        # Applying stereo image rectification on the right image
        Right_nice = cv2.remap(imgR_gray,
                               Right_Stereo_Map_x,
                               Right_Stereo_Map_y,
                               cv2.INTER_LANCZOS4,
                               cv2.BORDER_CONSTANT,
                               0)

        # Updating the parameters based on the trackbar positions
        # numDisparities_blockSize_preFilterType_preFilterSize_preFilterCap_textureThreshold_uniquenessRatio_speckleRange_speckleWindowSize_disp12MaxDiff_minDisparity

        # Setting the updated parameters before computing disparity map

        # Calculating disparity using the StereoBM algorithm
        disparity = stereo.compute(Left_nice, Right_nice)
        # NOTE: Code returns a 16bit signed single channel image,
        # CV_16S containing a disparity map scaled by 16. Hence it
        # is essential to convert it to CV_32F and scale it down 16 times.

        # Converting to float32
        disparity = disparity.astype(np.float32)

        # Scaling down the disparity values and normalizing them
        disparity = (disparity/16.0 - minDisparity)/numDisparities

        cv2.imwrite(output_folder+f"/disp_{frame_count}.png", disparity)
