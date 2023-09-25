import os

import cv2

from removeHndsUtils import remove_hands_from_image


def split_video_to_images(video_path, output_dir):
    os.system(f'ns-process-data video --data {video_path} --output-dir {output_dir} --skip-colmap')


def preprocess_data(input_dir, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # List all files in the input directory
    file_list = os.listdir(input_dir)

    # Loop over each file
    for file_name in file_list:
        # Check if the file is a PNG file
        if file_name.endswith('.png'):
            # Construct the full path of the input file
            input_path = os.path.join(input_dir, file_name)

            processed_image = remove_hands_from_image(input_path)

            # Construct the full path of the output file
            output_path = os.path.join(output_dir, file_name)

            # Save the modified image
            cv2.imwrite(output_path, processed_image)


def run_colmap_on_processed_images(images_path, output_dir):
    os.system(
        f'ns-process-data images --data {images_path} --output-dir {output_dir}')


def run_nerf(data_path):
    os.system(f'ns-train nerfacto --viewer.websocket-port 7007 nerfstudio-data --data {data_path} --downscale-factor 4')


if __name__ == '__main__':
    scene = 'matro'
    split_video_to_images(video_path=f"data\{scene}\{scene}.mp4", output_dir=f"data\{scene}")
    preprocess_data(input_dir=f'data/{scene}/images',
                    output_dir=f'data/{scene}/processed_images')
    run_colmap_on_processed_images(images_path=f'data/{scene}/processed_images', output_dir=f"data\{scene}")
    run_nerf(f"data\{scene}")
