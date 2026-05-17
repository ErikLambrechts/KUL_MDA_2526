import requests
import pandas as pd
import io
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Part 1

# Base configuration
base_url = 'https://opendata.apps.mow.vlaanderen.be/fietstellingen/'
start_date = datetime(2019, 8, 1)
end_date = datetime.now()
output_file = 'combined_fietstellingen_data.csv'

# Define requested column names
column_names = ['site_id', 'direction', 'type', 'from', 'to', 'amount']

# Remove existing file if we are restarting
if os.path.exists(output_file):
    os.remove(output_file)

current_date = start_date
first_file = True

print(f"Starting download process. Saving to {output_file}...")

while current_date <= end_date:
    month_str = current_date.strftime('%Y-%m')
    file_name = f'data-{month_str}.csv'
    file_url = f"{base_url}{file_name}"

    try:
        response = requests.get(file_url, timeout=30)
        if response.status_code == 200:
            # Read file with no header and assign names
            df = pd.read_csv(io.StringIO(response.text), header=None, names=column_names)
            df['source_month'] = month_str

            if first_file:
                # Write with headers for the first time
                df.to_csv(output_file, index=False, mode='w')
                first_file = False
                print(f"Created: {file_name}")
            else:
                # Append without headers
                df.to_csv(output_file, index=False, mode='a', header=False)
                print(f"Appended: {file_name}")

            del df
        else:
            pass
    except Exception as e:
        print(f"Error processing {file_name}: {e}")

    current_date += relativedelta(months=1)

print("\nProcess complete with custom headers.")


# Part 2:

# Load the directions metadata
directions_url = 'https://opendata.apps.mow.vlaanderen.be/fietstellingen/richtingen.csv'
directions = pd.read_csv(directions_url, header=None, names=['site_id', 'direction', 'direction_name'])

# Load the main dataset
# Note: Loading the entire file might be memory intensive, but necessary for the merge.
main_df = pd.read_csv('combined_fietstellingen_data.csv')

# Perform the merge
# This adds 'direction_name' to the main dataframe based on matching site_id and direction
merged_df = pd.merge(main_df, directions, on=['site_id', 'direction'], how='left')

# Save the new enriched dataset
final_output = 'final_combined_data.csv'
merged_df.to_csv(final_output, index=False)

print(f"Successfully merged data. New file saved as: {final_output}")
display(merged_df.head())


# Part 3:

# URL for site metadata
sites_url = 'https://opendata.apps.mow.vlaanderen.be/fietstellingen/sites.csv'

# Define the column names as requested
site_columns = [
    'site_id', 'site_nr', 'longitude', 'latitude', 'name_site',
    'domain', 'path_nr', 'district', 'council', 'interval', 'installation_date'
]

# Read the sites CSV
sites = pd.read_csv(sites_url, header=None, names=site_columns)

# Merge with the previously merged_df (which already contains directions)
# We use 'left' join on site_id to preserve all records in the main dataset
final_merged_df = pd.merge(merged_df, sites, on='site_id', how='left')

print("Merge with site metadata complete.")
display(final_merged_df.head())


# Part 4:

# Export total combined df
# Define the final path
drive_path = '/content/drive/My Drive/MDA Group Project/final_combined_data.csv'  #here need to add your path

# Save the merged dataframe to the Drive location
final_merged_df.to_csv(drive_path, index=False)
