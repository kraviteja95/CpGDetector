import os

# Specify the directory path where your app.py file is located
directory_path = '/Users/ravkothu/dev/flask_poc/CpG_Detector/bin'

# Specify the file name
file_name = 'cpg_detector_model.pth'

# Construct the full file path
file_path = os.path.join(directory_path, file_name)

# Check if the file exists
if os.path.exists(file_path):
    print(f"The file {file_name} exists in the directory.")
else:
    print(f"The file {file_name} does not exist in the directory.")