import imageio
import os
from tqdm import tqdm

def create_video_from_images(image_folder, output_video_path, frame_rate=25):
    # define valid extension
    valid_extensions = [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
    
    # get all image files in the folder
    image_files = [f for f in os.listdir(image_folder) 
                   if os.path.splitext(f)[1] in valid_extensions]
    image_files.sort()  # sort the files in alphabetical order
    print(image_files)
    if not image_files:
        raise ValueError("No valid image files found in the specified folder.")
    
    # create a video writer using imageio
    writer = imageio.get_writer(output_video_path, fps=frame_rate, codec='libx264', quality=8)
    
    # write each image to the video
    for image_file in tqdm(image_files):
        image_path = os.path.join(image_folder, image_file)
        image = imageio.imread(image_path)
        writer.append_data(image)
    
    # release the writer
    writer.close()
    print(f"Video saved at {output_video_path}")
