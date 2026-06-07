# conveer the mp4 video in mp3 audio 

import os 
import subprocess
files = os.listdir("videos")
# print(files)

for file in files:
    # print(file)
    tutorial_number = file.split("#")[1].split(".")[0]
    file_name = file.split("_")[0]
    print(f"{tutorial_number} {file_name}")
    subprocess.run(["ffmpeg", "-i", f"videos/{file}",f"audios/{tutorial_number}_{file_name}.mp3"])
    print("Congrates! \n Your all videos are converted successfully in audio.mp3")
    