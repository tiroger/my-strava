#############
# LIBRARIES #
#############

from dotenv import load_dotenv
import os

import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import json
import pandas as pd

#############################
# FUNCTION TO RETREIVE DATA #
#############################



def my_data():
    page = 1
    frames = []
    try:
        print('Getting data...')
        while page < 5: # Iterating through pages
            auth_url = "https://www.strava.com/oauth/token"
            activites_url = "https://www.strava.com/api/v3/athlete/activities"

            payload = {
                'client_id': os.getenv('CLIENT_ID'),
                'client_secret': os.getenv('CLIENT_SECRET'),
                'refresh_token': os.getenv('REFRESH_TOKEN'),
                'grant_type': "refresh_token",
                'f': 'json'
            }

            # print("Requesting Token...\n")
            res = requests.post(auth_url, data=payload, verify=False)
            print(res)
            access_token = res.json()['access_token']
            # print("Access Token = {}\n".format(access_token))

            header = {'Authorization': 'Bearer ' + access_token}
            param = {'per_page': 200, 'page': page}
            my_dataset = requests.get(activites_url, headers=header, params=param).json()
            output = pd.DataFrame(my_dataset) # Converting json to datarame
            frames.append(output)

            page += 1

            my_dataset_df = pd.concat(frames) # Concatenating all pages into dataframe
            print('Data retrieved successfully!')
            return my_dataset_df
            
    except:
        print('There was a problem with the request.')


def process_data(all_activities):
    #all_activities = my_data()

    # Transforming the date column to datetime and extracting year, month, weekday
    all_activities['start_date_local'] = pd.to_datetime(all_activities['start_date_local'])
    all_activities['year'] = all_activities['start_date_local'].dt.year
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    all_activities['month'] = all_activities['start_date_local'].dt.month
    all_activities['month'] = all_activities['month'].apply(lambda x: months[x-1])
    all_activities['day'] = all_activities['start_date_local'].dt.day
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    all_activities['weekday'] = all_activities['start_date_local'].dt.weekday
    all_activities['weekday'] = all_activities['weekday'].apply(lambda x: days[x]) # Converting weekday number to name
    all_activities['hour'] = all_activities['start_date_local'].dt.hour

    # Dropping unnecessary columns
    cols_to_remove = ['name', 'athlete', 'resource_state', 'upload_id_str', 'external_id', 
    'from_accepted_tag', 'has_kudoed', 'workout_type', 'display_hide_heartrate_option', 'map', 'visibility',
    'timezone', 'upload_id', 'start_date', 'start_date_local', 'utc_offset', 'location_city', 'location_country', 
    'location_state', 'heartrate_opt_out', 'flagged', 'commute', 'manual', 'gear_id', 'athlete_count', 'private', 
    'has_heartrate', 'start_latlng', 'end_latlng', 'device_watts', 'elev_low']
    activities_df = all_activities.drop(cols_to_remove, axis=1)

    return activities_df

if __name__ == '__main__':
    my_data()