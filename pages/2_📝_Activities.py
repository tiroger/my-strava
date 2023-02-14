#############
# LIBRARIES #
#############

from get_strava_data import my_data, process_data, bike_data, get_elev_data_GOOGLE # Functions to retrive data using strava api and process for visualizations

# import ast
# import polyline

import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
import datetime as dt

# # from PIL import Image
# import base64

# import matplotlib.pyplot as plt
# import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# import folium
# from folium.features import CustomIcon
# from streamlit_folium import folium_static

# import matplotlib.pyplot as plt

import streamlit as st
import streamlit.components.v1 as components

###############
# CREDENTIALS #
###############

token = MAPBOX_TOKEN = st.secrets['MAPBOX_TOKEN']
GOOGLE_API_KEY = st.secrets['GOOGLE_API_KEY']


###########################
# Main Page Configuration #
###########################

st.set_page_config(page_title='My Strava', 
                    page_icon='./icons/cropped-rtc-favicon.png', 
                    layout="wide", 
                    initial_sidebar_state="auto")

strava_color_palette = ['#FC4C02', '#45738F', '#DF553B', '#3A18B0', '#FFAA06', '#26A39E', '#951B05', '#D22B0C', '#F5674E'] # [strava orange, strava blue, warm orange, advance indigo, intermediate gold, beginner teal, hard red, medium sienna, easy peach]

# Strava widgets
latest_activities = 'https://www.strava.com/athletes/644338/latest-rides/53c89f1acdf2bf69a8bc937e3793e9bfb56d64be'
my_week = 'https://www.strava.com/athletes/644338/activity-summary/53c89f1acdf2bf69a8bc937e3793e9bfb56d64be'

#################
# DATE ELEMENTS #
#################

today = dt.datetime.today()
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
this_month = dt.datetime.today().month
this_year = dt.datetime.today().year


#################
# PERSONAL DATA #
#################

ftp = 192
weight_lbs = 164
weight_kg = weight_lbs * 0.453592

bikes_dict = {'Tie Fighter': 'Storck Scenero', 'Caadie': 'Cannondale CAAD10', 'Dirty McDirtBag': 'Marin Headlands', 'Hillius Maximus': 'Giant TCR', 'Hurt Enforcer': 'Cannondale Slate'}



#################################
# ACQUIRING AND PROCESSING DATA #
#################################


##### Use one of two options below #####

# Get data using strava api # For deployment

@st.cache(show_spinner=False, max_entries=5, ttl=86400, allow_output_mutation=True)
def fetch_activities():
    with st.spinner('Data Refreshing...'):

        my_data_df = my_data()
        processed_data = process_data(my_data_df)

        return processed_data

processed_data = fetch_activities()



# # Get local data # For development
# processed_data = pd.read_csv('./data/processed_data.csv')
# bikes_df = pd.read_csv('./data/bike_data.csv')
# athlete_df = pd.read_csv('./data/athlete_data.csv')



processed_data['start_date_local'] = pd.to_datetime(processed_data['start_date_local'])
processed_data['start_date_local'] = processed_data['start_date_local'].dt.strftime('%m-%d-%Y')

# Saving processed data to csv for future use
processed_data.to_csv('./data/processed_data.csv', index=False)



# Data for dahsboard
start_date = processed_data.year.min()
burger_calories = 354
total_activities = processed_data.id.count()
num_rides = (processed_data.type == 'Ride').sum()
num_runs = (processed_data.type == 'Workout').sum()
distance_traveled = processed_data.distance.sum().astype(int)
feet_climbed = processed_data.total_elevation_gain.sum().astype(int)
total_kudos = processed_data.kudos_count.sum()
earth_circumference = 24901 # earth circumference in miles 
perc_around_the_earth = (distance_traveled / earth_circumference)
total_time = processed_data.moving_time.sum()





####################
# Activities Table #
####################

st.markdown('<h2 style="color:#45738F">Activities</h2>', unsafe_allow_html=True)

# Filter by activity type
activity_type = st.selectbox('Filter by sport', ['Ride', 'VirtualRide', 'Run']) # Select from dropdown

sort_preference = st.radio('Sort by', ('Date', 'Distance (mi)', 'Elevation Gain (ft)', 'Elevation Gain/mile (ft)', 'Avg Speed (mph)', 'Avg Power (Watts)', 'Avg Heartrate', 'Suffer Score'))

# Processing data for table
streamlit_df = processed_data[['start_date_local', 'name', 'type', 'moving_time', 'distance', 'total_elevation_gain', 'elev_gain_per_mile', 'average_speed', 'average_cadence', 'average_watts', 'average_heartrate', 'suffer_score']]
streamlit_df['start_date_local'] = pd.to_datetime(streamlit_df['start_date_local'])
streamlit_df['start_date_local'] = streamlit_df['start_date_local'].dt.strftime('%Y-%m-%d')
streamlit_df.rename(columns={'start_date_local': 'Date','name': 'Name', 'type': 'Type', 'moving_time': 'Moving Time (h)', 'distance': 'Distance (mi)', 'total_elevation_gain': 'Elevation Gain (ft)', 'elev_gain_per_mile': 'Elevation Gain/mile (ft)', 'average_speed': 'Avg Speed (mph)', 'average_cadence': 'Avg Cadence (rpm)', 'average_watts': 'Avg Power (Watts)', 'average_heartrate': 'Avg Heartrate', 'suffer_score': 'Suffer Score'}, inplace=True)
#streamlit_df.set_index('Date', inplace=True)
streamlit_df = streamlit_df[streamlit_df['Type'].isin([activity_type])]

# Sorting table
streamlit_df.sort_values(by=sort_preference, ascending=False, inplace=True)

headerColor = '#45738F'
rowEvenColor = 'lightcyan'
rowOddColor = 'white'

# Plotly table
fig = go.Figure(data=[go.Table(
    columnorder = [1,2,3,4,5,6,7,8,9,10,11,12],
    columnwidth = [25,50,18,20,20,23,25,20,24,20,25,17],
    header=dict(values=list(streamlit_df.columns),
                line_color='darkslategray',
                fill_color=headerColor,
    font=dict(color='white', size=13)),
    cells=dict(values=[streamlit_df['Date'], streamlit_df['Name'], streamlit_df['Type'], streamlit_df['Moving Time (h)'], streamlit_df['Distance (mi)'], streamlit_df['Elevation Gain (ft)'], streamlit_df['Elevation Gain/mile (ft)'], streamlit_df['Avg Speed (mph)'], streamlit_df['Avg Cadence (rpm)'], streamlit_df['Avg Power (Watts)'], streamlit_df['Avg Heartrate'], streamlit_df['Suffer Score']],
               fill_color = [[rowOddColor,rowEvenColor]*len(streamlit_df.index),], font=dict(color='black', size=12), height=50))
])
fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))

# st.dataframe(streamlit_df)
st.plotly_chart(fig, use_container_width = True, config=dict(displayModeBar = False))



#################
 # MAP OF RIDES #
 ################

st.markdown('<h2 style="color:#45738F">Ride Maps</h2>', unsafe_allow_html=True)

polylines_df = pd.read_csv('./data/processed_data.csv', usecols=['start_date_local', 'name', 'distance', 'average_speed', 'total_elevation_gain', 'weighted_average_watts', 'average_heartrate', 'suffer_score', 'year', 'month', 'day', 'type', 'map'])
polylines_df.start_date_local = pd.DatetimeIndex(polylines_df.start_date_local)
polylines_df.start_date_local = polylines_df.start_date_local.dt.strftime('%m-%d-%Y')
polylines_df = polylines_df[polylines_df.type == 'Ride'] # We'll only use rides which have a map


option = st.selectbox(
     'Select a ride for more details',
     (polylines_df.name)
)

# Finding dataframe index based on ride name
idx = polylines_df[polylines_df.name == option].index.values[0]

# Decoding polylines
# Setting a try block in case a ride does not have a map
try:
    decoded = pd.json_normalize(polylines_df[polylines_df.index == idx]['map'].apply(ast.literal_eval))['summary_polyline'].apply(polyline.decode).values[0]


    # Adding elevation data from Google Elevation API
    @st.cache(persist=True, suppress_st_warning=True)

    def elev_profile_chart():
        with st.spinner('Calculating elevation profile from Google Elevation. Hang tight...'):
            elevation_profile_feet = [get_elev_data_GOOGLE(coord[0], coord[1]) for coord in decoded]
            # elevation_profile_feet = [elevation_profile[i] * 3.28084 for i in range(len(elevation_profile))] # Converting elevation to feet

            return elevation_profile_feet


    elevation_profile_feet = elev_profile_chart()

except:
    st.write('Geocoordinates are unavailable for this activity')