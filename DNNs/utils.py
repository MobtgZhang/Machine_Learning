import imageio
import os
def create_gif(image_list, gif_name):
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(image_name))
    # Save them as frames into a gif 
    imageio.mimsave(gif_name, frames, 'GIF', duration = 0.1)
def save_gif(file_path,gif_name):
    image_list = []
    for f in os.listdir(file_path):
        image_list.append(os.path.join(file_path,f))
    create_gif(image_list, gif_name)