#############
# LIBRARIES #
#############

# from dotenv import load_dotenv
# load_dotenv()
import os
import ast

import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import json
import pandas as pd

import streamlit as st

# Credentials
# CLIENT_ID = st.secrets['CLIENT_ID']
# CLIENT_SECRET = st.secrets['CLIENT_SECRET']
# REFRESH_TOKEN = st.secrets['REFRESH_TOKEN']
# GOOGLE_API_KEY = st.secrets['GOOGLE_API_KEY']

CLIENT_ID = os.environ['CLIENT_ID']
CLIENT_SECRET = os.environ['CLIENT_SECRET']
REFRESH_TOKEN = os.environ['REFRESH_TOKEN']
GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']

#################################################
# FUNCTIONS TO RETRIEVE AND PROCESS STRAVA DATA #
#################################################


def my_data():

    try:
        page = 1
        frames = []
        print('Getting data...')
        while page < 10: # Iterating through all pages
            print(f"Requesting page {page}...")
            auth_url = "https://www.strava.com/oauth/token"
            activites_url = "https://www.strava.com/api/v3/athlete/activities"

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
            if my_dataset == []:
                break
            output = pd.DataFrame(my_dataset) # Converting json to datarame

            frames.append(output)
            
            page = page + 1 # Incrementing page number

            my_dataset_df = pd.concat(frames) # Concatenating all pages into dataframe
            
        print('Data retrieved successfully!')
        return my_dataset_df
            
    except:
        print('There was a problem with the request.')


import requests
import pandas as pd
import pickle
from datetime import datetime
import os

# Function to fetch new activities
import requests
import pandas as pd
import os

def fetch_new_activities():
    # Initialize existing_data
    existing_data = pd.DataFrame()

    # Load existing data if available
    if os.path.exists('my_dataset_df.feather'):
        existing_data = pd.read_feather('my_data.feather')
        last_activity_time = existing_data['start_date'].max()  # Assuming 'start_date' contains the activity start datetime
        last_activity_epoch = int(pd.to_datetime(last_activity_time).timestamp())
    else:
        last_activity_epoch = 0  # This will fetch all activities if no data exists

    try:
        # Check for the most recent activity on the server before making full API calls
        auth_url = "https://www.strava.com/oauth/token"
        activities_url = "https://www.strava.com/api/v3/athlete/activities"
        payload = {
            'client_id': CLIENT_ID,
            'client_secret': CLIENT_SECRET,
            'refresh_token': REFRESH_TOKEN,
            'grant_type': "refresh_token",
            'f': 'json'
        }

        # Authenticate and get the access token
        res = requests.post(auth_url, data=payload, verify=False)
        access_token = res.json()['access_token']

        # Request only the most recent activity
        header = {'Authorization': 'Bearer ' + access_token}
        param = {'per_page': 1, 'page': 1}
        recent_activity = requests.get(activities_url, headers=header, params=param).json()

        # Check if there is new activity
        if recent_activity and pd.to_datetime(recent_activity[0]['start_date']).timestamp() > last_activity_epoch:
            # New activity exists, proceed to fetch all new activities
            page = 1
            frames = []
            while page <= 10:  # Limiting to 10 pages for safety
                param = {'per_page': 200, 'page': page, 'after': last_activity_epoch}
                my_dataset = requests.get(activities_url, headers=header, params=param).json()
                
                if not my_dataset:
                    break  # Exit loop if no more data is returned

                output = pd.DataFrame(my_dataset)
                frames.append(output)
                page += 1

            new_activities_df = pd.concat(frames) if frames else pd.DataFrame()
            print('New activities fetched successfully!')

            # Combine with existing data, if available
            if not existing_data.empty:
                updated_data = pd.concat([existing_data, new_activities_df]).drop_duplicates().reset_index(drop=True)
            else:
                updated_data = new_activities_df
            
            # Save the updated data
            updated_data.reset_index().to_feather('my_data.feather')
            return updated_data
        else:
            # No new activities, return existing data
            print('No new activities found.')
            return existing_data

    except Exception as e:
        print(f'There was a problem with the request: {e}')
        return None


def refresh_access_token():
    auth_url = "https://www.strava.com/oauth/token"
    payload = {
            'client_id': CLIENT_ID,
            'client_secret': CLIENT_SECRET,
            'refresh_token': REFRESH_TOKEN,
            'grant_type': "refresh_token",
            'f': 'json'
    }
    res = requests.post(auth_url, data=payload)
    return res.json()['access_token']



def fetch_activity_streams(activity_id):
    try:
        # Authenticate and get the access token
        auth_url = "https://www.strava.com/oauth/token"
        payload = {
            'client_id': CLIENT_ID,
            'client_secret': CLIENT_SECRET,
            'refresh_token': REFRESH_TOKEN,
            'grant_type': 'refresh_token',
        }
        res = requests.post(auth_url, data=payload, verify=False)  # You should handle verification in production
        if res.status_code != 200:
            print(f"Failed to retrieve access token: {res.json()}")
            return None
        access_token = res.json()['access_token']

        # Define the URL for the Strava activity streams endpoint
        streams_url = f"https://www.strava.com/api/v3/activities/{activity_id}/streams"
        params = {
            'keys': 'time,distance,latlng,altitude,velocity_smooth,heartrate,cadence,watts,temp',
            'key_by_type': 'true'  # This organizes the response by stream type
        }
        headers = {'Authorization': f'Bearer {access_token}'}

        # Make the API request for the activity streams
        response = requests.get(streams_url, headers=headers, params=params)
        if response.status_code != 200:
            print(f"Failed to retrieve activity streams: {response.json()}")
            return None

        # Process the response
        streams_data = response.json()
        data_dict = {}

        for k, v in streams_data.items():
            data_dict[k] = v['data']
        
        df = pd.DataFrame(data_dict)
        return df

    except Exception as e:
        print(f'There was a problem with the request: {e}')
        return None


def athlete_data():
    print(f"Requesting data...")
    auth_url = "https://www.strava.com/oauth/token"
    athlete_url = 'https://www.strava.com/api/v3/athlete'

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

    # Adding a column for elevation gain per mile
    all_activities['elev_gain_per_mile'] = (all_activities['total_elevation_gain'] / all_activities['distance']).round(1)

    # Dropping unnecessary columns
    cols_to_remove = ['athlete', 'resource_state', 'upload_id_str', 'external_id', 
    'from_accepted_tag', 'has_kudoed', 'workout_type', 'display_hide_heartrate_option', 'visibility',
    'timezone', 'start_date', 'utc_offset', 'location_city', 'location_country', 
    'location_state', 'heartrate_opt_out', 'flagged', 'commute', 'manual', 'athlete_count', 'private', 
    'has_heartrate', 'start_latlng', 'end_latlng', 'device_watts', 'elev_low']
    activities_df = all_activities.drop(cols_to_remove, axis=1)

    return activities_df

######################################################
# FUNCTION TO RETRIEVE ELEVATION DATA FROM OPEN MAPS #
######################################################

def get_elevation(latitude, longitude):
    base_url = 'https://api.open-elevation.com/api/v1/lookup'
    payload = {'locations': f'{latitude},{longitude}'}
    r = requests.get(base_url, params=payload).json()['results'][0]
    return r['elevation']

###################################################
# FUNCTION TO RETRIEVE ELEVATION DATA FROM GOOGLE #
###################################################

def get_elev_data_GOOGLE(lat, lon):
    url = f"https://maps.googleapis.com/maps/api/elevation/json?locations={lat}%2C{lon}&key={GOOGLE_API_KEY}"

    payload={}
    headers = {}

    response = requests.request("GET", url, headers=headers, data=payload)
    response = ast.literal_eval(response.text)
    elevation = response['results'][0]['elevation']
    elevation_feet = elevation * 3.28084

    return elevation_feet


if __name__ == '__main__':
    my_data()