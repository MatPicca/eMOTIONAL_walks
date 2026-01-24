import os
import pandas as pd
import torch
from PIL import Image
# from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
# from transformers import AutoProcessor, Mask2FormerForUniversalSegmentation
#from transformers import Mask2FormerForUniversalSegmentation
# from transformers.processing_utils import AutoProcessor
from transformers.models.mask2former import Mask2FormerForUniversalSegmentation
from transformers.models.auto.processing_auto import AutoProcessor
from tqdm import tqdm
import numpy as np


def process_image(image_path, processor, model, class_descriptions):
    try:
        image = Image.open(image_path)
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        predicted_map = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]

        unique_classes, counts = torch.unique(predicted_map, return_counts=True)
        total_pixels = counts.sum().item()
        features = {label: 0 for label in set(class_descriptions.values())}

        for cls, count in zip(unique_classes.numpy(), counts.numpy()):
            if cls in class_descriptions:
                label = class_descriptions[cls]
                features[label] += count.item()

        features = {k: (v / total_pixels) * 100 for k, v in features.items()}
        return features

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None  # Returns None to indicate processing failure

def process_directory(directory, processor, model, class_descriptions):
    results = []
    image_files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for filename in image_files:
        image_path = os.path.join(directory, filename)
        features = process_image(image_path, processor, model, class_descriptions)
        if features is not None:  # Ignores images with errors
            results.append(features)

    if results:
        feature_df = pd.DataFrame(results)
        mean_features = feature_df.mean().to_dict()
        return mean_features
    else:
        print(f"No valid images found in directory {directory}")
        return {}  # Returns an empty dictionary if no valid image was processed


def process_images_in_batches(input_dir, dest_dir, batch_size=500, 
                              progress_file="/mnt/raid/matteo/Output_csv_mapillary_segmentation/processed_folders.txt"):
    # processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-mapillary-vistas-semantic")
    # processor = AutoProcessor.from_pretrained("facebook/mask2former-swin-large-mapillary-vistas-semantic")
    processor = AutoProcessor.from_pretrained("facebook/mask2former-swin-large-mapillary-vistas-semantic")

    model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-mapillary-vistas-semantic")
    class_descriptions = {
        13: "Road", 24: "Lane Marking - General", 41: "Manhole",
        2: "Sidewalk", 15: "Curb",
        17: "Building", 6: "Wall", 3: "Fence",
        45: "Pole", 47: "Utility Pole",
        48: "Traffic Light", 50: "Traffic Sign (Front)",
        30: "Vegetation", 29: "Terrain", 27: "Sky",
        19: "Person", 20: "Bicyclist", 21: "Motorcyclist", 22: "Other Rider",
        55: "Car", 61: "Truck", 54: "Bus", 58: "On Rails", 57: "Motorcycle", 52: "Bicycle"
    }

    # Load already processed folders
    processed_folders = set()
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            processed_folders.update(f.read().splitlines())

    all_dirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    remaining_dirs = [d for d in all_dirs if d not in processed_folders]
    dir_batches = np.array_split(remaining_dirs, max(1, len(remaining_dirs) // batch_size))

    for i, batch in enumerate(tqdm(dir_batches, desc="Processing batches")):
        aggregated_results = []
        for coord in tqdm(batch, desc=f"Processing batch {i + 1}"):
            coord_path = os.path.join(input_dir, coord)
            mean_features = process_directory(coord_path, processor, model, class_descriptions)
            mean_features['GRID_ID'] = coord  # Add GRID_ID to the row
            aggregated_results.append(mean_features)

            # Log progress after processing each folder
            with open(progress_file, "a") as f:
                f.write(f"{coord}\n")

        # Save batch results
        results_df = pd.DataFrame(aggregated_results)
        os.makedirs(dest_dir, exist_ok=True)
        batch_result_path = os.path.join(dest_dir, f'aggregated_results_batch_{i + 1}.csv')
        results_df.to_csv(batch_result_path, index=False)
        print(f"Batch {i + 1} results saved to {batch_result_path}")


if __name__ == "__main__":
    input_dir = r"/mnt/raid/matteo/Mapillary_images/" # Directory containing subfolders of images
    dest_dir = r"/mnt/raid/matteo/Output_csv_mapillary_segmentation" # Directory containing the output CSV files
    process_images_in_batches(input_dir, dest_dir, batch_size=50)
