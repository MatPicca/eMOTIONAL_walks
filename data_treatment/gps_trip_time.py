'''
Docstring for data_treatment.gps_trip_time
This script processes raw GPS data and matches it with walking trip diary data to create a comprehensive dataset of raw GPS points associated with specific trip (Inteval ID).
It reads the final foot trajectory data, filters raw GPS data for each participant, and performs a conditional join to match GPS points with trip intervals.
The resulting dataset is saved as a CSV file for further analysis.
'''

import pandas as pd
import subprocess
import os
import geopandas as gpd

import site 
# print(site.getusersitepackages()) # check site package location
import sys
sys.path = [
    '/home/s232713/.local/lib/python3.10/site-packages'
] + sys.path # force the path to include janitor location
import janitor
print(janitor.__file__)
sys.path.insert(0, '/home/s232713/.local/lib/python3.10/site-packages')
from janitor import conditional_join
from tqdm import tqdm

personal_id = xxxxxxxxxxx # Replace xxxxxxxxxxx with your actual personal ID as an integer

# take the final foot trajectory data used to create traj and inside the boundaries
df_foot = pd.read_pickle('/home/s232713/data/trajectories/FINAL_foot_data.pkl') 
print(len(df_foot), df_foot['Interval ID'].nunique()) # 2018 # 2018
print(df_foot['INDIVID'].nunique()) # 110
gps_path = f'/run/user/{personal_id}/gvfs/smb-share:server=ait-pdfs.win.dtu.dk,share=department/Man/Public/4233-81647-eMOTIONAL-Cities/5 Data/ECDTU/Xing/GPS/Final/'

# Get unique participant IDs from the trips data
mmm_id = df_foot.INDIVID.unique()

all_trips = []

for subj in tqdm(mmm_id, desc="Processing participants"):
    print('PROCESSING INDIVIDUAL:', subj, '.....')
    # Construct the command to filter the raw GPS data for the specific participant
    command = "awk -F'\\t' 'NR==1 || $1~/" + str(subj) + "/' raw.csv > out2matteo.csv"
    # Execute the command in the specified directory
    subprocess.Popen(command, cwd=gps_path, shell=True).wait()

    # Define the path to the output file
    out2_path = os.path.join(gps_path, 'out2matteo.csv')
    # Check if the output file is not empty
    if os.path.getsize(out2_path) > 0:
        # Read the filtered GPS data into a DataFrame
        raw_gps_data_subj = pd.read_csv(out2_path, sep=',')
    else:
        # Create an empty DataFrame if the file is empty
        raw_gps_data_subj = pd.DataFrame()

    # Remove the temporary output file
    os.remove(out2_path)

    # Filter the diary data for the specific participant
    diary_sub = df_foot[df_foot["INDIVID"] == subj]

    # Convert columns to datetime
    raw_gps_data_subj['Timestamp'] = pd.to_datetime(raw_gps_data_subj['Timestamp'], utc=True)
    raw_gps_data_subj["Timestamp"] = raw_gps_data_subj["Timestamp"].dt.tz_convert("Europe/Copenhagen")

    # Convert '(TZ-Aware)' in xing_data to datetime
    raw_gps_data_subj['Timestamp'] = raw_gps_data_subj['Timestamp'].dt.tz_localize(None)

    # raw_gps_data_subj.head(5)

    print('Performing conditional join for individual:', subj)
    # Perform the conditional join
    trips = conditional_join(
        raw_gps_data_subj,
        diary_sub,
        ('Timestamp', 'Start Time_x', '>='), 
        ('Timestamp', 'End Time_x', '<=') 
    )
    trips = trips[[
        ('left', 'INDIVID'),
        ('left', 'Timestamp'),
        ('left', 'Latitude'),
        ('left', 'Longitude'),
        ('left', 'Accuracy'),
        ('left', 'Altitude'),
        ('left', 'Speed'),
        ('right', 'Interval ID'),
        ('right', 'Activity_concat')
    ]]

    # Remove the 'left' and 'right' prefixes from the column names
    trips.columns = trips.columns.droplevel(0)

    # Convert 'Interval ID' column to integer
    trips['Interval ID'] = trips['Interval ID'].astype(int)

    trips['Milliseconds'] = (trips['Timestamp'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1ms')

    all_trips.append(trips)


all_trips_df = pd.concat(all_trips, ignore_index=True)

print(all_trips_df.head()) 
print('number of individual id: ', all_trips_df['INDIVID'].nunique())
print('number of interval id: ', all_trips_df['Interval ID'].nunique())

save_path = '/home/s232713/data/trip_time.csv'
all_trips_df.to_csv(save_path)
print('Data saved to:', save_path)


