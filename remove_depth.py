import os

# Replace with the path to your folder
folder_path = "/Users/tarioyou/Downloads/nerf-master/drum_test_no_depth"

# Iterate over all files in the folder
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)

    # Check if the file doesn't contain the word "depth" in its filename
    if "depth" in filename:
        # Delete the file
        os.remove(file_path)
        print(f"Deleted file: {filename}")
