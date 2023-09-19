from os import listdir
from os.path import isfile, join
import os
import shutil
import sys

old_path = sys.argv[1]
new_path = sys.argv[2]

# Ensure the new path exists; create it if it doesn't
if not os.path.exists(new_path):
    os.makedirs(new_path)

for folder in os.listdir(old_path):
    folder_path = join(old_path, folder)
    
    if os.path.isdir(folder_path):
        for file in listdir(folder_path):
            file_path = join(folder_path, file)
            
            if isfile(file_path):
                filename, file_extension = os.path.splitext(file)
                parts = filename.split('-')
                
                if len(parts) >= 3:
                    emotion_folder = join(new_path, parts[2])
                    
                    if not os.path.exists(emotion_folder):
                        os.makedirs(emotion_folder)
                    
                    destination = join(emotion_folder, file)
                    
                    try:
                        shutil.move(file_path, destination)
                    except Exception as e:
                        print(f"Error moving file {file}: {e}")
