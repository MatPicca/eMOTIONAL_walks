import pandas as pd
import os
import requests
from tqdm import tqdm
from math import isnan, radians, cos, log, sin, sqrt, atan2
import torch
import logging

import time
import random
import itertools

############### used screen on python terminal to run it background and avoid disconnections ###############

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#------ create the hexgrid centroid file with only the hex touched by trajectories ------
hex_csv = pd.read_csv('/home/s232713/data/grid_data/cph_hexgrid_centroids.csv', sep=',')
grid_trip_csv = pd.read_csv('/home/s232713/data/grid_data/grid_to_trip.csv', sep=',')
merged = grid_trip_csv.merge(
    hex_csv[['GRID_ID', 'lat', 'lon']],
    on='GRID_ID',
    how='left'  # keeps only the rows from grid_trip_csv, adds lat/lon if found
)
# Keep only these three columns
final = merged[['GRID_ID', 'lat', 'lon']].drop_duplicates()
#-----------------------------------------------------------------------------------------

full_df = final # DataFrame with lat, lon, GRID_ID columns
# input_file = final # Input CSV file with lat, lon, GRID_ID columns 
output_dir = '/mnt/raid/matteo/POI/POI_traj_touched/' # Directory to save output CSV files
radius = 50  # Search radius in meters
block_size = 1000 # number of rows to process in each block

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

os.makedirs(output_dir, exist_ok=True)

# Rotate between multiple Overpass servers to reduce throttling
OVERPASS_SERVERS = itertools.cycle([
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter"
])

'''OVERPASS_SERVERS = itertools.cycle([
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.nchc.org.tw/api/interpreter",     
    "https://overpass.openstreetmap.ru/api/interpreter"  
])''' # to make it even more robust if needed but maybe too many servers for this project

def query_overpass(lat, lon, radius, max_retries=3):
    """Query Overpass API with retry, delay, and mirror rotation."""
    overpass_query = f"""
        [out:json][timeout:25];
        (
          node(around:{radius},{lat},{lon});
          way(around:{radius},{lat},{lon});
          rel(around:{radius},{lat},{lon});
        );
        out body;
        >;
        out skel qt;
    """

    for attempt in range(max_retries):
        server = next(OVERPASS_SERVERS)
        time.sleep(0.5 + random.random() * 0.5)  # base delay to slow requests a bit

        try:
            response = requests.get(server, params={'data': overpass_query})
            response.raise_for_status()
            return response.json()

        except requests.exceptions.HTTPError as e:
            # Handle rate limit (429) and timeout (504)
            if e.response.status_code == 429:
                wait = 5 + random.random() * 7 # random wait between 5 and 12 seconds
                logging.warning(f"⚠️  Rate limited (429). Waiting {wait:.1f}s before retry...")
                time.sleep(wait)
            elif e.response.status_code == 504:
                wait = 3 + random.random() * 5 # random wait between 3 and 8 secondst
                logging.warning(f"⚠️  Timeout (504). Retrying after {wait:.1f}s...")
                time.sleep(wait)
            else:
                raise
        except Exception as e:
            logging.warning(f"⚠️  Query failed ({e}). Retrying...")
            time.sleep(3 + random.random() * 5) # random wait between 3 and 8 seconds

    logging.error(f"❌  Failed after {max_retries} attempts for ({lat},{lon}).")
    return {"elements": []}


def categorize_and_count(elements, categories):
    counts = {category: 0 for category in categories.values()}
    for element in elements:
        tags = element.get('tags', {})
        for key, value in tags.items():
            category = categories.get(f"{key}:{value}")
            if category:
                counts[category] += 1
    return counts

def calculate_shannon_diversity_index(counts, device):
    total = sum(counts.values())
    if total == 0:
        return 0

    counts_tensor = torch.tensor([count for count in counts.values()], dtype=torch.float32, device=device)
    total_tensor = torch.tensor(total, dtype=torch.float32, device=device)
    proportions = counts_tensor / total_tensor
    log_proportions = torch.log(proportions)
    shannon_diversity = -torch.sum(proportions * log_proportions)

    return shannon_diversity.item() if not isnan(shannon_diversity.item()) else 0

'''def get_max_elevation(lat, lon, distance):
    directions = [(lat + distance / 111111, lon),  # North
                  (lat - distance / 111111, lon),  # South
                  (lat, lon + distance / (111111 * cos(radians(lat)))),  # East
                  (lat, lon - distance / (111111 * cos(radians(lat))))]  # West

    max_elevation = None
    for direction in directions:
        try:
            response = query_overpass(direction[0], direction[1], distance)
            elements = response.get('elements', [])
            for element in elements:
                elevation = float(element.get('tags', {}).get('ele', 0))
                if max_elevation is None or elevation > max_elevation:
                    max_elevation = elevation
        except Exception as e:
            logging.warning(f"Erro ao buscar elevação para direção {direction}: {e}")
            continue

    return max_elevation'''

def process_block(df, output_path, categories, device):
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        lat, lon, grid_id = row['lat'], row['lon'], row['GRID_ID']
        try:
            response = query_overpass(lat, lon, radius)
            elements = response.get('elements', [])
            counts = categorize_and_count(elements, categories)
            diversity_index = calculate_shannon_diversity_index(counts, device)

            '''current_elevation_data = response['elements']
            current_elevation = float(
                current_elevation_data[0].get('tags', {}).get('ele', 0)) if current_elevation_data else 0
            max_elevation = get_max_elevation(lat, lon, 50)
            slope = (max_elevation - current_elevation) if max_elevation else 0'''
            slope = 0  # Placeholder since elevation data gives problems
            ### need to fix it somehow (eliminate it or find another source)

            results.append({
                'GRID_ID': grid_id,
                'lat': lat,
                'lon': lon,
                'Diversity_Index': diversity_index,
                'Slope': slope,
                **counts
            })
        except Exception as e:
            logging.warning(f"Erro ao processar GRID_ID {grid_id}: {e}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, sep=',', index=False)

# full_df = pd.read_csv(input_file, sep=',') # use it if input_file is a path
num_blocks = (len(full_df) + block_size - 1) // block_size

processed_ids = set()
progress_file = os.path.join(output_dir, 'progresso.txt')

if os.path.exists(progress_file):
    with open(progress_file, 'r') as f:
        processed_ids.update(line.strip() for line in f)

categories = {
    # Urban Vibrancy
        "amenity:arts_centre": "poi_amenity",
        "amenity:community_centre": "poi_amenity",
        "amenity:fountain": "poi_amenity",
        "amenity:library": "poi_amenity",
        "amenity:marketplace": "poi_amenity",
        "amenity:place_of_worship": "poi_amenity",
        "amenity:public_bookcase": "poi_amenity",
        "amenity:social_centre": "poi_amenity",
        "amenity:theatre": "poi_amenity",
        "amenity:townhall": "poi_amenity",
        "amenity:atm": "poi_amenity",
        "amenity:bank": "poi_amenity",
        "amenity:bar": "poi_amenity",
        "amenity:bicycle_rental": "poi_amenity",
        "amenity:biergarten": "poi_amenity",
        "amenity:buddhist": "poi_amenity",
        "amenity:cafe": "poi_amenity",
        "amenity:car_rental": "poi_amenity",
        "amenity:car_wash": "poi_amenity",
        "amenity:christian": "poi_amenity",
        "amenity:cinema": "poi_amenity",
        "amenity:clinic": "poi_amenity",
        "amenity:college": "poi_amenity",
        "amenity:courthouse": "poi_amenity",
        "amenity:dentist": "poi_amenity",
        "amenity:doctors": "poi_amenity",
        "amenity:fast_food": "poi_amenity",
        "amenity:fire_station": "poi_amenity",
        "amenity:food_court": "poi_amenity",
        "amenity:graveyard": "poi_amenity",
        "amenity:hospital": "poi_amenity",
        "amenity:kindergarten": "poi_amenity",
        "amenity:market_place": "poi_amenity",
        "amenity:nightclub": "poi_amenity",
        "amenity:pharmacy": "poi_amenity",
        "amenity:police": "poi_amenity",
        "amenity:post_office": "poi_amenity",
        "amenity:pub": "poi_amenity",
        "amenity:public_building": "poi_amenity",
        "amenity:restaurant": "poi_amenity",
        "amenity:school": "poi_amenity",
        "amenity:toilet": "poi_amenity",
        "amenity:university": "poi_amenity",
        "amenity:veterinary": "poi_amenity",
        "shop:bakery": "poi_shop",
        "shop:beauty_shop": "poi_shop",
        "shop:beverages": "poi_shop",
        "shop:bicycle_shop": "poi_shop",
        "shop:bookshop": "poi_shop",
        "shop:butcher": "poi_shop",
        "shop:car_dealership": "poi_shop",
        "shop:chemist": "poi_shop",
        "shop:clothes": "poi_shop",
        "shop:computer_shop": "poi_shop",
        "shop:convenience": "poi_shop",
        "shop:department_store": "poi_shop",
        "shop:florist": "poi_shop",
        "shop:furniture_shop": "poi_shop",
        "shop:garden_centre": "poi_shop",
        "shop:gift_shop": "poi_shop",
        "shop:greengrocer": "poi_shop",
        "shop:hairdresser": "poi_shop",
        "shop:jeweller": "poi_shop",
        "shop:kiosk": "poi_shop",
        "shop:laundry": "poi_shop",
        "shop:mall": "poi_shop",
        "shop:mobile_phone_shop": "poi_shop",
        "shop:newsagent": "poi_shop",
        "shop:optician": "poi_shop",
        "shop:outdoor_shop": "poi_shop",
        "shop:shoe_shop": "poi_shop",
        "shop:sports_shop": "poi_shop",
        "shop:stationery": "poi_shop",
        "shop:supermarket": "poi_shop",
        "shop:toy_shop": "poi_shop",
        "shop:travel_agent": "poi_shop",
        "amenity:bar": "poi_food",
        "amenity:cafe": "poi_food",
        "amenity:fast_food": "poi_food",
        "amenity:pub": "poi_food",
        "amenity:restaurant": "poi_food",
        "amenity:bicycle_parking": "poi_transportation",
        "amenity:bus_station": "poi_transportation",
        "amenity:car_rental": "poi_transportation",
        "amenity:charging_station": "poi_transportation",
        "amenity:ferry_terminal": "poi_transportation",
        "amenity:fuel": "poi_transportation",
        "amenity:parking": "poi_transportation",
        "amenity:taxi": "poi_transportation",
        "leisure:garden": "poi_leisure",
        "leisure:nature_reserve": "poi_leisure",
        "leisure:park": "poi_leisure",
        "leisure:stadium": "poi_leisure",
        "leisure:playground": "poi_leisure",
        "leisure:dog_park": "poi_leisure",
        "leisure:pitch": "poi_leisure",
        "leisure:sports_centre": "poi_leisure",
        "leisure:track": "poi_leisure",
        "tourism:artwork": "poi_tourism",
        "tourism:attraction": "poi_tourism",
        "tourism:gallery": "poi_tourism",
        "tourism:museum": "poi_tourism",
        "tourism:viewpoint": "poi_tourism",
        "tourism:zoo": "poi_tourism",
        "natural:tree": "poi_natural",
        "natural:waterfall": "poi_natural",
        "natural:beach": "poi_natural",
        "natural:peak": "poi_natural",
        "building:apartments": "poi_buiding",
        "building:commercial": "poi_buiding",
        "building:hotel": "poi_buiding",
        "building:industrial": "poi_buiding",
        "building:retail": "poi_buiding",
        "building:house": "poi_buiding",
        "historic:castle": "poi_historic",
        "historic:memorial": "poi_historic",
        "historic:monument": "poi_historic",
        "historic:statue": "poi_historic",
        "historic:ruins": "poi_historic",
        "historic:archaeological_site": "poi_historic",
        "tourism:guesthouse": "poi_historic",
        "tourism:hostel": "poi_historic",
        "tourism:hotel": "poi_historic",
        "tourism:picnic_site": "poi_historic",
        "tourism:theme_park": "poi_historic",
        "tourism:information": "poi_historic",
        "tourism:viewpoint": "poi_historic",
        "historic:wayside_shrine": "poi_historic",
        "highway:footway": "poi_highway",
        "highway:cycleway": "poi_highway",
        "highway:living_street": "poi_highway",
        "highway:pedestrian": "poi_highway",
        "highway:residential": "poi_highway",
        "highway:steps": "poi_highway",
        "highway:track": "poi_highway",
        ################################################ Lines Tags for Catgeories ################################################
        # Food and Drink
        "amenity:restaurant": "food_drink",
        "amenity:cafe": "food_drink",
        "amenity:bar": "food_drink",
        "amenity:pub": "food_drink",
        "amenity:biergarten": "food_drink",
        "amenity:fast_food": "food_drink",
        "amenity:food_court": "food_drink",
        "amenity:ice_cream": "food_drink",
        "amenity:marketplace": "food_drink",
        "shop:supermarket": "food_drink",
        "shop:department_store": "food_drink",
        "shop:mall": "food_drink",
        "shop:clothes": "food_drink",
        "shop:alcohol": "food_drink",
        "shop:bakery": "food_drink",
        "shop:beverages": "food_drink",
        "shop:brewing_supplies": "food_drink",
        "shop:butcher": "food_drink",
        "shop:cheese": "food_drink",
        "shop:chocolate": "food_drink",
        "shop:coffee": "food_drink",
        "shop:confectionery": "food_drink",
        "shop:convenience": "food_drink",
        "shop:deli": "food_drink",
        "shop:dairy": "food_drink",
        "shop:farm": "food_drink",
        "shop:frozen_food": "food_drink",
        "shop:greengrocer": "food_drink",
        "shop:health_food": "food_drink",
        "shop:ice_cream": "food_drink",
        "shop:organic": "food_drink",
        "shop:pasta": "food_drink",
        "shop:pastry": "food_drink",
        "shop:seafood": "food_drink",
        "shop:spices": "food_drink",
        "shop:tea": "food_drink",
        "shop:wine": "food_drink",
        "shop:water": "food_drink",
        "shop:department_store": "food_drink",
        "shop:general": "food_drink",
        "shop:kiosk": "food_drink",
        # Buildings and Facilities
        "building:commercial": "buildings_facilities",
        "building:residential": "buildings_facilities",
        "building:house": "buildings_facilities",
        "building:apartments": "buildings_facilities",
        "building:retail": "buildings_facilities",
        "building:industrial": "buildings_facilities",
        "building:warehouse": "buildings_facilities",
        "building:school": "buildings_facilities",
        "building:university": "buildings_facilities",
        "building:college": "buildings_facilities",
        "building:kindergarten": "buildings_facilities",
        "amenity:hospital": "buildings_facilities",
        "amenity:police": "buildings_facilities",
        "amenity:fire_station": "buildings_facilities",
        "amenity:library": "buildings_facilities",
        "amenity:community_centre": "buildings_facilities",
        "amenity:school": "buildings_facilities",
        "amenity:university": "buildings_facilities",
        "amenity:college": "buildings_facilities",
        "amenity:kindergarten": "buildings_facilities",
        "amenity:clinic": "buildings_facilities",
        "amenity:doctors": "buildings_facilities",
        "amenity:dentist": "buildings_facilities",
        "amenity:veterinary": "buildings_facilities",
        "leisure:pitch": "buildings_facilities",
        "leisure:stadium": "buildings_facilities",
        "leisure:sports_centre": "buildings_facilities",
        "leisure:swimming_pool": "buildings_facilities",
        "office:it": "buildings_facilities",
        "office:software": "buildings_facilities",
        "office:research": "buildings_facilities",
        "amenity:bureau_de_change": "buildings_facilities",
        "amenity:money_transfer": "buildings_facilities",
        "amenity:shelter": "buildings_facilities",
        "amenity:social_facility": "buildings_facilities",
        "amenity:social_centre": "buildings_facilities",
        # Roads and Transportation
        "highway:motorway": "roads_transportation",
        "highway:motorway_link": "roads_transportation",
        "highway:trunk": "roads_transportation",
        "highway:trunk_link": "roads_transportation",
        "highway:primary": "roads_transportation",
        "highway:primary_link": "roads_transportation",
        "highway:secondary": "roads_transportation",
        "highway:secondary_link": "roads_transportation",
        "highway:tertiary": "roads_transportation",
        "highway:tertiary_link": "roads_transportation",
        "highway:residential": "roads_transportation",
        "highway:living_street": "roads_transportation",
        "highway:service": "roads_transportation",
        "highway:unclassified": "roads_transportation",
        "highway:road": "roads_transportation",
        "highway:footway": "roads_transportation",
        "highway:pedestrian": "roads_transportation",
        "highway:path": "roads_transportation",
        "highway:cycleway": "roads_transportation",
        "highway:bridleway": "roads_transportation",
        "highway:steps": "roads_transportation",
        "highway:track": "roads_transportation",
        "highway:busway": "roads_transportation",
        "highway:construction": "roads_transportation",
        "highway:proposed": "roads_transportation",
        "highway:escape": "roads_transportation",
        "highway:emergency_bay": "roads_transportation",
        "highway:crossing": "roads_transportation",
        "highway:turning_circle": "roads_transportation",
        "highway:turning_loop": "roads_transportation",
        "highway:mini_roundabout": "roads_transportation",
        "highway:traffic_signals": "roads_transportation",
        "highway:stop": "roads_transportation",
        "highway:give_way": "roads_transportation",
        "highway:services": "roads_transportation",
        "highway:rest_area": "roads_transportation",
        "highway:speed_camera": "roads_transportation",
        "highway:toll_booth": "roads_transportation",
        "highway:milestone": "roads_transportation",
        "highway:passing_place": "roads_transportation",
        "highway:bus_stop": "roads_transportation",
        "highway:street_lamp": "roads_transportation",
        # Tourism and Leisure
        "tourism:hotel": "roads_transportation",
        "tourism:motel": "tourism_leisure",
        "tourism:guest_house": "tourism_leisure",
        "tourism:hostel": "tourism_leisure",
        "tourism:bed_and_breakfast": "tourism_leisure",
        "tourism:camp_site": "tourism_leisure",
        "tourism:caravan_site": "roads_transportation",
        "tourism:chalet": "roads_transportation",
        "tourism:information": "roads_transportation",
        "tourism:museum": "roads_transportation",
        "tourism:gallery": "roads_transportation",
        "tourism:zoo": "roads_transportation",
        "tourism:theme_park": "roads_transportation",
        "tourism:attraction": "roads_transportation",
        "tourism:viewpoint": "roads_transportation",
        "tourism:picnic_site": "roads_transportation",
        "leisure:park": "roads_transportation",
        "leisure:garden": "roads_transportation",
        "leisure:nature_reserve": "roads_transportation",
        "leisure:bowling_alley": "roads_transportation",
        "leisure:arcade": "roads_transportation",
        "leisure:playground": "roads_transportation",
        "amenity:cinema": "roads_transportation",
        "amenity:gym": "roads_transportation",
        "amenity:theatre": "roads_transportation",
        "amenity:arts_centre": "roads_transportation",
        "amenity:studio": "roads_transportation",
        "amenity:nightclub": "roads_transportation",
        "amenity:casino": "roads_transportation",
        "amenity:stripclub": "roads_transportation",
        "amenity:event_venue": "roads_transportation",
        "tourism:festival": "roads_transportation",
        "tourism:party_venue": "roads_transportation",
        # Greenery and Natural Features
        "landuse:forest": "greenery_natural",
        "landuse:grass": "greenery_natural",
        "landuse:orchard": "greenery_natural",
        "landuse:farmland": "greenery_natural",
        "landuse:allotments": "greenery_natural",
        "landuse:recreation_ground": "greenery_natural",
        "landuse:vineyard": "greenery_natural",
        "natural:wood": "greenery_natural",
        "natural:tree": "greenery_natural",
        "natural:tree_row": "greenery_natural",
        "natural:scrub": "greenery_natural",
        "natural:heath": "greenery_natural",
        "natural:grassland": "greenery_natural",
        "natural:moor": "greenery_natural",
        "natural:wetland": "greenery_natural",
        "natural:mud": "greenery_natural",
        "natural:land": "greenery_natural",
        "amenity:dog_park": "greenery_natural",
        "amenity:picnic_site": "greenery_natural",
        "amenity:park": "greenery_natural",
        "amenity:garden": "greenery_natural",
        "amenity:nature_reserve": "greenery_natural",
        # Public Services and Facilities
        "amenity:place_of_worship": "public_services",
        "amenity:post_office": "public_services",
        "amenity:bank": "public_services",
        "amenity:atm": "public_services",
        "amenity:pharmacy": "public_services",
        "amenity:prison": "public_services",
        "amenity:lawyer": "public_services",
        "amenity:notary": "public_services",
        "amenity:townhall": "public_services",
        "amenity:courthouse": "public_services",
        "amenity:embassy": "public_services",
        "amenity:government": "public_services",
        "amenity:telephone": "public_services",
        "amenity:internet_cafe": "public_services",
        "amenity:newsagent": "public_services",
        "amenity:waste_basket": "public_services",
        "amenity:waste_disposal": "public_services",
        "amenity:waste_transfer_station": "public_services",
        "amenity:water_fountain": "public_services",
        # Arts, Entertainment, and Events
        "amenity:studio": "arts_entertainment_events",
        "amenity:arts_centre": "arts_entertainment_events",
        "amenity:nightclub": "arts_entertainment_events",
        "amenity:casino": "arts_entertainment_events",
        "amenity:stripclub": "arts_entertainment_events",
        "amenity:event_venue": "arts_entertainment_events",
        "tourism:festival": "arts_entertainment_events",
        "tourism:party_venue": "arts_entertainment_events",
        # Beauty, Personal Care, and Health
        "amenity:beauty_salon": "beauty_personal_health",
        "amenity:hairdresser": "beauty_personal_health",
        "amenity:nail_salon": "beauty_personal_health",
        "amenity:clinic": "beauty_personal_health",
        "amenity:doctors": "beauty_personal_health",
        "amenity:dentist": "beauty_personal_health",
        "amenity:veterinary": "beauty_personal_health",
        # Miscellaneous and Other Services
        "amenity:social_facility": "miscellaneous_services",
        "amenity:social_centre": "miscellaneous_services",
        "amenity:coworking_space": "miscellaneous_services",
        "office:it": "miscellaneous_services",
        "office:software": "miscellaneous_services",
        "office:research": "miscellaneous_services",
        "amenity:bureau_de_change": "miscellaneous_services",
        "amenity:money_transfer": "miscellaneous_services",
        "amenity:shelter": "miscellaneous_services",
        "amenity:recycling": "miscellaneous_services",
        "amenity:theatre": "miscellaneous_services",
        "amenity:newsagent": "miscellaneous_services",
        # Water Bodies
        "natural:water": "water_body",
        "natural:lake": "water_body",
        "natural:river": "water_body",
        "natural:stream": "water_body",
        "natural:pond": "water_body",
        "waterway:river": "water_body",
        "waterway:stream": "water_body",
        "waterway:canal": "water_body",
        "waterway:drain": "water_body",
        "waterway:ditch": "water_body"
}

for i in range(num_blocks):
    start = i * block_size
    end = (i + 1) * block_size
    block_df = full_df.iloc[start:end]
    block_ids = set(block_df['GRID_ID'].astype(str))
    output_path = os.path.join(output_dir, f'block_{i + 1}.csv')

    if block_ids.issubset(processed_ids):
        logging.info(f'Bloco {i + 1} já processado. Pulando...')
        continue

    logging.info(f'Processando bloco {i + 1}/{num_blocks}...')
    process_block(block_df, output_path, categories, device)

    with open(progress_file, 'a') as f:
        for grid_id in block_ids:
            f.write(f'{grid_id}\n')

logging.info('Processamento concluído.')
