import requests
import os
import pandas as pd
import shutil
import json
from vt2geojson.tools import vt_bytes_to_geojson
import mercantile
from tqdm import tqdm
from geopy.distance import geodesic


#------ create the hexgrid centroid file with only the hex touched by trajectories ------
hex_csv = pd.read_csv('/home/s232713/data/grid_data/cph_hexgrid_centroids.csv', sep=',') # columns: GRID_ID, lat, lon
grid_trip_csv = pd.read_csv('/home/s232713/data/grid_data/grid_to_trip.csv', sep=',') # columns: GRID_ID, TRIP_ID
merged = grid_trip_csv.merge(
    hex_csv[['GRID_ID', 'lat', 'lon']],
    on='GRID_ID',
    how='left'  # keeps only the rows from grid_trip_csv, adds lat/lon if found
)
# Keep only these three columns
final = merged[['GRID_ID', 'lat', 'lon']].drop_duplicates()
#-----------------------------------------------------------------------------------------


# Constants
TILE_COVERAGE = 'mly1_public'
TILE_LAYER = "image"
ACCESS_TOKEN = 'MLY|24943991905228322|90e9a23a0f494d3da57e2bfdb8c4b7cc'  # Mapillary access token

# Set variables directly in the script
dest_dir = '/mnt/raid/matteo/Mapillary_images/'  # image saved in the external raid
# dest_dir = '/home/s232713/data/Mapillary_images/'  # Directory to save images and metadata
#input_file = 'C:/Dados_GPS/Data.csv'  # Path to the CSV file with Latitude and Longitude (Hexagon cell centroid)
image_size = 1024  # Image size (320, 640, 1024, 2048)
n_images = 10  # Number of images per location
radius = 50  # Search radius in meters
compressed_output = True  # Defines whether the result will be compressed into a ZIP

# Functions
def create_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)

def download_images():
    df = final.rename(columns={'lat': 'Latitude', 'lon': 'Longitude'}) # Use the prepared DataFrame
    #df = pd.read_csv(input_file, sep=";") # this line if you read from a CSV file instead (uncomment line 32 input_file='PATH')
    metadata = []
    coordinates_with_images = []
    coordinates_without_images = []

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Downloading images", unit="coordinate"):
        lat, lon = row['Latitude'], row['Longitude']
        grid_id = row['GRID_ID']  # Get the GRID_ID
        coord_folder = f"{grid_id}"  # Use GRID_ID as folder name

        # skip grid if folder already exists
        coord_dir = os.path.join(dest_dir, coord_folder)
        if os.path.exists(coord_dir):
            print(f"Skipping GRID_ID {grid_id} (folder already exists).")
            continue

        images_downloaded = download_and_save(lat, lon, metadata, coord_folder)

        if images_downloaded > 0:
            coordinates_with_images.append({'GRID_ID': grid_id, 'Latitude': lat, 'Longitude': lon})
        else:
            coordinates_without_images.append({'GRID_ID': grid_id, 'Latitude': lat, 'Longitude': lon})

        print(f"Downloaded {images_downloaded} images for GRID_ID {grid_id}.")


def download_and_save(lat, lon, metadata, coord_folder):
    coord_dir = os.path.join(dest_dir, coord_folder)
    create_dir(coord_dir)
    tiles = list(mercantile.tiles(lon, lat, lon, lat, 14))  # Zoom level 14
    images_collected = 0

    for tile in tiles:
        if images_collected >= n_images:
            break
        tile_url = f'https://tiles.mapillary.com/maps/vtp/{TILE_COVERAGE}/2/{tile.z}/{tile.x}/{tile.y}?access_token={ACCESS_TOKEN}'
        response = requests.get(tile_url)
        if response.ok:
            data = vt_bytes_to_geojson(response.content, tile.x, tile.y, tile.z, layer=TILE_LAYER)
            for feature in data['features']:
                feature_id = feature['properties']['id']
                feature_lat = feature['geometry']['coordinates'][1]
                feature_lon = feature['geometry']['coordinates'][0]
                if geodesic((lat, lon), (feature_lat, feature_lon)).meters <= radius:
                    if images_collected < n_images:
                        save_image(feature_id, lat, lon, metadata, coord_folder)
                        images_collected += 1
                    else:
                        break
    return images_collected


def save_image(image_id, lat, lon, metadata, coord_folder):
    coord_dir = os.path.join(dest_dir, coord_folder)
    image_url = f'https://graph.mapillary.com/{image_id}?fields=thumb_{image_size}_url&access_token={ACCESS_TOKEN}'
    response = requests.get(image_url)
    
    if response.ok:
        image_data = response.json()
        
        # Check if the specified size field is present
        image_key = f'thumb_{image_size}_url'
        if image_key in image_data:
            filename = f"{image_id}.jpg"
            image_path = os.path.join(coord_dir, filename)
            
            # Download and save the image
            try:
                image_content = requests.get(image_data[image_key]).content
                with open(image_path, 'wb') as file:
                    file.write(image_content)
                metadata.append({
                    "image_id": image_id,
                    "latitude": lat,
                    "longitude": lon,
                    "grid_id": coord_folder,
                    "file_path": image_path
                })
            except Exception as e:
                print(f"Error saving image {image_id}: {e}")
        else:
            # Missing field: log and continue
            print(f"Image {image_id} does not have the specified size: {image_key}")
    else:
        print(f"API response error when fetching image URL for {image_id}: {response.status_code}")

if __name__ == "__main__":
    # Start the image download process
    download_images()
    print("Image download process completed.")



