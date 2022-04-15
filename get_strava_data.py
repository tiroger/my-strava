#############
# LIBRARIES #
#############

from dotenv import load_dotenv
load_dotenv()
import os

import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import json
import pandas as pd

import streamlit as st

# For Streamlit
CLIENT_ID = st.secrets['CLIENT_ID']
CLIENT_SECRET = st.secrets['CLIENT_SECRET']
REFRESH_TOKEN = st.secrets['REFRESH_TOKEN']

# For local development
# CLIENT_ID = os.environ('CLIENT_ID')
# CLIENT_SECRET = os.environ('CLIENT_SECRET')
# REFRESH_TOKEN = os.environ('REFRESH_TOKEN')

#############################
# FUNCTION TO RETREIVE DATA #
#############################



def my_data():

    try:
        page = 1
        frames = []
        print('Getting data...')
        while page < 5: # Iterating through all pages
            print(f"Requesting page {page}...")
            auth_url = "https://www.strava.com/oauth/token"
            activites_url = "https://www.strava.com/api/v3/athlete/activities"

            # payload = {
            #     'client_id': os.getenv('CLIENT_ID'),
            #     'client_secret': os.getenv('CLIENT_SECRET'),
            #     'refresh_token': os.getenv('REFRESH_TOKEN'),
            #     'grant_type': "refresh_token",
            #     'f': 'json'
            # }
            # For stramlit
            payload = {
                'client_id': CLIENT_ID,
                'client_secret': CLIENT_SECRET,
                'refresh_token': REFRESH_TOKEN,
                'grant_type': "refresh_token",
                'f': 'json'
            }

            
            res = requests.post(auth_url, data=payload, verify=False)
            print(res)
            access_token = res.json()['access_token']
            # print("Access Token = {}\n".format(access_token))

            header = {'Authorization': 'Bearer ' + access_token}
            param = {'per_page': 200, 'page': page}
            my_dataset = requests.get(activites_url, headers=header, params=param).json()
            output = pd.DataFrame(my_dataset) # Converting json to datarame
            frames.append(output)
            
            page = page + 1 # Incrementing page number

            my_dataset_df = pd.concat(frames) # Concatenating all pages into dataframe
            
        print('Data retrieved successfully!')
        return my_dataset_df
            
    except:
        print('There was a problem with the request.')

def athlete_data():
    print(f"Requesting data...")
    auth_url = "https://www.strava.com/oauth/token"
    athlete_url = 'https://www.strava.com/api/v3/athlete'

    # payload = {
    #     'client_id': os.getenv('CLIENT_ID'),
    #     'client_secret': os.getenv('CLIENT_SECRET'),
    #     'refresh_token': os.getenv('REFRESH_TOKEN'),
    #     'grant_type': "refresh_token",
    #     'f': 'json'
    # }
    # For stramlit
    payload = {
        'client_id': st.secrets('CLIENT_ID'),
        'client_secret': st.secrets('CLIENT_SECRET'),
        'refresh_token': st.secrets('REFRESH_TOKEN'),
        'grant_type': "refresh_token",
        'f': 'json'
    }
    
    res = requests.post(auth_url, data=payload, verify=False)
    print(res)
    access_token = res.json()['access_token']
    # print("Access Token = {}\n".format(access_token))

    header = {'Authorization': 'Bearer ' + access_token}
    param = {'per_page': 200, 'page': 1}
    athlete = requests.get(athlete_url, headers=header, params=param).json()
    df = pd.json_normalize(athlete)
    # athlete_df = pd.DataFrame(athlete)
    
    print('Data retrieved successfully!')
    # return athlete_df
    return df

def bike_data():
    try:
        bikes = ['b8099416', 'b4196400', 'b8615449', 'b4073790', 'b5245627', 'b8029179', 'b326351', 'b804798', 'b232108']
        frames = []
        for b in bikes:
            print('Requesting data...')
            auth_url = "https://www.strava.com/oauth/token"
            gears_url = f'https://www.strava.com/api/v3/gear/{b}'
            
            # payload = {
            # 'client_id': os.getenv('CLIENT_ID'),
            # 'client_secret': os.getenv('CLIENT_SECRET'),
            # 'refresh_token': os.getenv('REFRESH_TOKEN'),
            # 'grant_type': "refresh_token",
            # 'f': 'json'
            # }
            
            # For stramlit
            payload = {
                'client_id': st.secrets('CLIENT_ID'),
                'client_secret': st.secrets('CLIENT_SECRET'),
                'refresh_token': st.secrets('REFRESH_TOKEN'),
                'grant_type': "refresh_token",
                'f': 'json'
            }

            res = requests.post(auth_url, data=payload, verify=False)
            print(res)
            access_token = res.json()['access_token']
            # print("Access Token = {}\n".format(access_token))

            header = {'Authorization': 'Bearer ' + access_token}
            param = {'per_page': 200, 'page': 1}
            gears = requests.get(gears_url, headers=header).json()
            df = pd.json_normalize(gears) # Converting json to datarame
            frames.append(df)
            bikes_df = pd.concat(frames) # Concatenating all pages into dataframe
            print('Data retrieved successfully!')
        return bikes_df
    except:
        print('There was a problem with the request.')


def process_data(all_activities):

    # Transforming the date column to datetime and extracting year, month, weekday
    all_activities['start_date_local'] = pd.to_datetime(all_activities['start_date_local'])
    all_activities['year'] = all_activities['start_date_local'].dt.year
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    all_activities['month'] = all_activities['start_date_local'].dt.month
    # all_activities['month'] = all_activities['month'].apply(lambda x: months[x-1])
    all_activities['day'] = all_activities['start_date_local'].dt.day
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    all_activities['weekday'] = all_activities['start_date_local'].dt.weekday
    all_activities['weekday'] = all_activities['weekday'].apply(lambda x: days[x]) # Converting weekday number to name
    all_activities['hour'] = all_activities['start_date_local'].dt.hour

    # Converting distance to miles
    all_activities['distance'] = (all_activities['distance'] * 0.000621371).round(1)

    # Converting speeds to mph
    all_activities['average_speed'] = (all_activities['average_speed'] * 2.23694).round(1)
    all_activities['max_speed'] = (all_activities['max_speed'] * 2.23694).round(1)

    # Converting elevation to feet
    all_activities['total_elevation_gain'] = (all_activities['total_elevation_gain'] * 3.28084).round(1)
    all_activities['elev_high'] = (all_activities['elev_high'] * 3.28084).round(1)

    # Converting elapse time to hours
    all_activities['elapsed_time'] = (all_activities['elapsed_time'] / 3600).round(1)
    all_activities['moving_time'] = (all_activities['moving_time'] / 3600).round(1)

    # Dropping unnecessary columns
    cols_to_remove = ['athlete', 'resource_state', 'upload_id_str', 'external_id', 
    'from_accepted_tag', 'has_kudoed', 'workout_type', 'display_hide_heartrate_option', 'map', 'visibility',
    'timezone', 'upload_id', 'start_date', 'utc_offset', 'location_city', 'location_country', 
    'location_state', 'heartrate_opt_out', 'flagged', 'commute', 'manual', 'athlete_count', 'private', 
    'has_heartrate', 'start_latlng', 'end_latlng', 'device_watts', 'elev_low']
    activities_df = all_activities.drop(cols_to_remove, axis=1)

    return activities_df

if __name__ == '__main__':
    my_data()