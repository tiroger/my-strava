#############
# LIBRARIES #
#############

from dotenv import load_dotenv
import os

import requests
import json
import pandas as pd

#############################
# FUNCTION TO RETREIVE DATA #
#############################



def my_data():
    page = 1
    frames = []
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
        access_token = res.json()['access_token']
        # print("Access Token = {}\n".format(access_token))

        header = {'Authorization': 'Bearer ' + access_token}
        param = {'per_page': 200, 'page': page}
        my_dataset = requests.get(activites_url, headers=header, params=param).json()
        output = pd.DataFrame(my_dataset) # Converting json to datarame
        frames.append(output)

        page += 1

        my_dataset_df = pd.concat(frames) # Concatenating all pages into dataframe

    return my_dataset_df