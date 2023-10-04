import os
import cv2
import shutil
from removeHndsUtils import remove_hands_from_image
from tqdm import tqdm


def prepare_data_for_colmap(input_path, output_dir, video_or_images):
    os.system(f'ns-process-data {video_or_images} --data {input_path} --output-dir {output_dir} --skip-colmap')


def preprocess_data(input_dir, output_dir, remove_bg):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    # List all files in the input directory
    file_list = os.listdir(input_dir)
    # Loop over each image
    for file_name in tqdm(file_list):
        if file_name.endswith('.png') or file_name.endswith('.jpg'):
            input_path = os.path.join(input_dir, file_name)
            # Remove hands from image
            processed_image = remove_hands_from_image(input_path, remove_bg)
            file_name = file_name.split(".")[0] + '.png'
            output_path = os.path.join(output_dir, file_name)
            cv2.imwrite(output_path, processed_image)


def run_colmap_on_processed_images(images_path, output_dir):
    os.system(
        f'ns-process-data images --data {images_path} --output-dir {output_dir}')


def run_nerf(data_path):
    os.system(f'ns-train nerfacto --viewer.websocket-port 7007 nerfstudio-data --data {data_path} --downscale-factor 16')


def remove_data(dir, processed_images_dir, base_dir):
    if os.path.exists(processed_images_dir):
        shutil.move(processed_images_dir, base_dir)
    if os.path.exists(dir):
        shutil.rmtree(dir)


if __name__ == '__main__':
    scene = 'rubic_up_1'
    video_format = 'mp4' #MOV/mp4
    if os.path.exists(f"data/{scene}/{scene}.{video_format}"):
        video_or_images = 'video'
        input_path = f"data/{scene}/{scene}.{video_format}"
    else:
        video_or_images = 'images'
        input_path = f"data/{scene}/images"
    if os.path.exists(f"data/{scene}/processed"):
        shutil.rmtree(f"data/{scene}/processed")
    prepare_data_for_colmap(input_path=input_path, output_dir=f"data/{scene}/processed/splitted_data",
                          video_or_images=video_or_images)
    preprocess_data(input_dir=f'data/{scene}/processed/splitted_data/images',
                    output_dir=f'data/{scene}/processed/splitted_data/processed_images', remove_bg=False)
    remove_data(f"data/{scene}/processed/splitted_data", f"data/{scene}/processed/splitted_data/processed_images",
                f"data/{scene}/processed")
    run_colmap_on_processed_images(images_path=f'data/{scene}/processed/processed_images',
                                   output_dir=f"data/{scene}/processed")
    run_nerf(f"data/{scene}/processed")
