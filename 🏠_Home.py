#############
# LIBRARIES #
#############

from get_strava_data import my_data, process_data, bike_data, get_elev_data_GOOGLE, fetch_new_activities # Functions to retrive data using strava api and process for visualizations

# import ast
# import polyline

import pyarrow.feather as feather

import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
import datetime as dt

# from PIL import Image
import base64

import matplotlib.pyplot as plt
# import seaborn as sns
import calplot
import plotly.express as px
import plotly.graph_objects as go

# import folium
# from folium.features import CustomIcon
# from streamlit_folium import folium_static

import streamlit as st
import streamlit.components.v1 as components

import pickle
import os
# from PIL import Image


###############
# CREDENTIALS #
###############

# MAPBOX_TOKEN = st.secrets['MAPBOX_TOKEN']
# GOOGLE_API_KEY = st.secrets['GOOGLE_API_KEY']

CLIENT_ID = os.environ['CLIENT_ID']
CLIENT_SECRET = os.environ['CLIENT_SECRET']
REFRESH_TOKEN = os.environ['REFRESH_TOKEN']
GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']
MAPBOX_TOKEN = os.environ['MAPBOX_TOKEN']

####################
# FOR IMAGES ICONS #
####################

@st.cache_data
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return


###########################
# Main Page Configuration #
###########################

st.set_page_config(page_title='My Strava', 
                    page_icon='./icons/cropped-rtc-favicon.png', 
                    layout="wide", 
                    initial_sidebar_state="auto")

strava_color_palette = ['#FC4C02', '#45738F', '#DF553B', '#3A18B0', '#FFAA06', '#26A39E', '#951B05', '#D22B0C', '#F5674E'] # [strava orange, strava blue, warm orange, advance indigo, intermediate gold, beginner teal, hard red, medium sienna, easy peach]

# # Strava widgets
# latest_activities = 'https://www.strava.com/athletes/644338/latest-rides/53c89f1acdf2bf69a8bc937e3793e9bfb56d64be'
# my_week = 'https://www.strava.com/athletes/644338/activity-summary/53c89f1acdf2bf69a8bc937e3793e9bfb56d64be'

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
weight_lbs = 152
weight_kg = weight_lbs * 0.453592

bikes_dict = {'Tie Fighter': 'Storck Scenero', 'Caadie': 'Cannondale CAAD10', 'Dirty McDirtBag': 'Marin Headlands', 'Hillius Maximus': 'Giant TCR', 'Hurt Enforcer': 'Cannondale Slate'}



#################################
# ACQUIRING AND PROCESSING DATA #
#################################


##### Use one of two options below #####

# Get data using strava api # For deployment

@st.cache_data(show_spinner=False, max_entries=5, ttl=43200)
def fetch_activities():
    with st.spinner('Data Refreshing...'):

        my_data_df = my_data()
        processed_data = process_data(my_data_df)

        return processed_data, my_data_df

@st.cache_data(show_spinner=False, max_entries=5, ttl=43200)
def bikes():
    with st.spinner('Data Refreshing...'):
        bikes = bike_data()

        return bikes

processed_data, my_data_df = fetch_activities()
bikes_df = bikes()

# Save the raw data as a feather file
feather.write_feather(my_data_df, 'my_data.feather')
# Last recorded activity -- use start_date_local
last_recorded_activity = my_data_df.start_date_local.max()
# st.write(f'Last recorded activity: {last_recorded_activity}')
with open('last_recorded_activity.pkl', 'wb') as f:
    pickle.dump(last_recorded_activity, f)



# Save the last activity time as a pickle file
# with open('last_activity_time.pkl', 'wb') as f:
#     pickle.dump(last_activity_time, f)
    

# Get local data # For development
# processed_data = pd.read_csv('./data/processed_data.csv')
# bikes_df = pd.read_csv('./data/bike_data.csv')
# athlete_df = pd.read_csv('./data/athlete_data.csv')

    #############################


processed_data['start_date_local'] = pd.to_datetime(processed_data['start_date_local'])
processed_data['start_date_local'] = processed_data['start_date_local'].dt.strftime('%m-%d-%Y')

# Saving processed data to csv for future use
processed_data.to_csv('./data/processed_data.csv', index=False)



# Data for dahsboard
start_date = processed_data.year.min()
burger_calories = 354
# total_activities = processed_data.id.count()
num_rides = (processed_data.type == 'Ride').sum()
num_runs = (processed_data.type == 'Workout').sum()
distance_traveled = processed_data.distance.sum().astype(int)
feet_climbed = processed_data.total_elevation_gain.sum().astype(int)
total_kudos = processed_data.kudos_count.sum()
earth_circumference = 24901 # earth circumference in miles 
perc_around_the_earth = (distance_traveled / earth_circumference)
total_time = processed_data.moving_time.sum()
total_kilojoules_burned = processed_data.kilojoules.sum()
# Convert kilojoules to calories
total_calories_burned = total_kilojoules_burned * 0.239005736



###############
# INFOGRAPHIC #
###############


############
# SIDE BAR #
############

# Adding option to select year and activity type in the sidebar
years_on_strava = processed_data.year.unique().tolist()
with st.sidebar:
    message = """
            __Select Year and Activity Type__
            """
    st.markdown(message, unsafe_allow_html=True)
    # st.subheader('Select Year and Activity Type')
    year = st.selectbox('Year', years_on_strava)
    # By default "All Activities" is selected
    selected_activities = []
    all_activities = st.checkbox('All Activities', value=True)
    if all_activities:
        selected_activities = ['Ride', 'VirtualRide', 'Run']
        # st.write(f'You selected the following activities: {selected_activities}')
    # If all activities is not selected, then allow user to select rides, virtual rides, or runs
    if not all_activities:
        rides = st.checkbox('Rides', value=False)
        if rides:
            selected_activities.append('Ride')
        virtual_rides = st.checkbox('Virtual Rides', value=False)
        if virtual_rides:
            selected_activities.append('VirtualRide')
        runs = st.checkbox('Runs', value=False)
        if runs:
            selected_activities.append('Run')

    
#############
# MAIN PAGE #
#############

new_title = f'<H2 style="font-family:sans-serif; color:#fc4c02; font-size: 48px;">STRAVA OVERVIEW FOR {year} </H2>'
st.markdown(new_title, unsafe_allow_html=True)
last_recorded_activity_human_readable = pd.to_datetime(last_recorded_activity).strftime('%B %d, %Y')
last_recorded_time_activity_human_readable = pd.to_datetime(last_recorded_activity).strftime('%I:%M %p')
st.write(f'Last recorded activity: {last_recorded_activity_human_readable} at {last_recorded_time_activity_human_readable}')


no_activity = 'No activity selected'

if st.button('Fetch New Activities'):
    new_data = fetch_new_activities()
    if new_data is not None:
        st.write('New activities fetched successfully!')
        # st.dataframe(new_data)
    else:
        st.write('No new activities or error in fetching data.')

# Columns for data
col1, col2, col3= st.columns(3)

# Total activities
with col1:
    activities_title =  f'<H2 style="font-family:sans-serif; font-size: 20px;"><i>Activities</i></H2>'
    st.markdown(activities_title, unsafe_allow_html=True)
    st.image('./icons/calendar.png', width=50)
    total_activities = total_activities = round(processed_data[(processed_data.year == year) & (processed_data.type.isin(selected_activities))].id.count(), 0).astype(int)
    total_activities = f'{total_activities:,}' # Formatted with commas
    # choice = 'activities'
    if selected_activities == ['Ride']:
        choice = 'rides'
    elif selected_activities == ['VirtualRide']:
        choice = 'virtual rides'
    elif selected_activities == ['Run']:
        choice = 'runs'
    elif selected_activities == []:
        choice = no_activity
        total_activities = ''
    else:
        choice = ''
    total_activities = f'<p style="font-family:sans-serif; color:#45738F; font-size: 42px;">{total_activities} {choice}</p>'
    st.markdown(total_activities, unsafe_allow_html=True) 

# Total distance traveled
with col2:
    distance_title =  f'<H2 style="font-family:sans-serif; font-size: 20px;"><i>Distance</i></H2>'
    st.markdown(distance_title, unsafe_allow_html=True)
    st.image('./icons/distance.png', width=50)
    total_distance = total_distance = round(processed_data[(processed_data.year == year) & (processed_data.type.isin(selected_activities))].distance.sum(), 0).astype(int)
    total_distance = f'{total_distance:,}' # Formatted with commas
    if selected_activities == []:
        distance_traveled = ''
    else:
        distance_traveled = f'<p style="font-family:sans-serif; color:#45738F; font-size: 42px;">{total_distance} miles</p>'
    st.markdown(distance_traveled, unsafe_allow_html=True)
    
# Total moving time
with col3:
    time_title =  f'<H2 style="font-family:sans-serif; font-size: 20px;"><i>Moving Time</i></H2>'
    st.markdown(time_title, unsafe_allow_html=True)
    st.image('./icons/stopwatch.png', width=50)
    total_moving_time = total_moving_time = round(processed_data[(processed_data.year == year) & (processed_data.type.isin(selected_activities))].moving_time.sum(), 2).astype(int)
    # # Converting to hours and minutes style
    # hours = int(total_moving_time)
    # minutes = (total_moving_time*60) % 60
    # seconds = (total_moving_time*3600) % 60
    # total_moving_time = f'{hours}h {minutes}m {seconds}s'
    if selected_activities == []:
        total_moving_time = ''
    else:
        total_moving_time = f'<p style="font-family:sans-serif; color:#45738F; font-size: 42px;">{total_moving_time} hours</p>'
    st.markdown(total_moving_time, unsafe_allow_html=True)
    
    
# Columns for data
col4, col5, col6= st.columns(3)

# Total elevation gain
with col4:
    elevation_title =  f'<H2 style="font-family:sans-serif; font-size: 20px;"><i>Elevation Gain</i></H2>'
    st.markdown(elevation_title, unsafe_allow_html=True)
    st.image('./icons/mountain.png', width=50)
    total_elevation_gain = total_elevation_gain = round(processed_data[(processed_data.year == year) & (processed_data.type.isin(selected_activities))].total_elevation_gain.sum(), 0).astype(int) 
    total_elevation_gain = f'{total_elevation_gain:,}' # Formatted with commas
    if selected_activities == []:
        total_elevation_gain = ''
    else:
        total_elevation_gain = f'<p style="font-family:sans-serif; color:#45738F; font-size: 42px;">{total_elevation_gain} feet</p>'
    st.markdown(total_elevation_gain, unsafe_allow_html=True)
    
# Total calories
with col5:
    calories_title =  f'<H2 style="font-family:sans-serif; font-size: 20px;"><i>Calories</i></H2>'
    st.markdown(calories_title, unsafe_allow_html=True)
    st.image('./icons/calories.png', width=50)
    total_calories_burned = total_calories_burned = round(processed_data[(processed_data.year == year) & (processed_data.type.isin(selected_activities))].kilojoules.sum(), 0).astype(int)
    # total_calories_burned = round(total_calories_burned  * 0.239005736, 0).astype(int)
    # To burgers
    # total_calories_burned = round(total_calories_burned / burger_calories, 0).astype(int)
    total_calories_burned = f'{total_calories_burned:,}' # Formatted with commas
    if selected_activities == []:
        total_calories_burned = ''
    else:
        total_calories_burned = f'<p style="font-family:sans-serif; color:#45738F; font-size: 42px;">{total_calories_burned} kcal</p>'
    st.markdown(total_calories_burned, unsafe_allow_html=True)

# Total kudos
with col6:
    kudos_title =  f'<H2 style="font-family:sans-serif; font-size: 20px;"><i>Likes</i></H2>'
    st.markdown(kudos_title, unsafe_allow_html=True)
    st.image('./icons/like.png', width=50)
    total_kudos = total_kudos = round(processed_data[(processed_data.year == year) & (processed_data.type.isin(selected_activities))].kudos_count.sum(), 0).astype(int)
    # total_calories_burned = round(total_calories_burned  * 0.239005736, 0).astype(int)
    # To burgers
    # total_calories_burned = round(total_calories_burned / burger_calories, 0).astype(int)
    total_kudos = f'{total_kudos:,}' # Formatted with commas
    if selected_activities == []:
        total_kudos = ''
    else:
        total_kudos = f'<p style="font-family:sans-serif; color:#45738F; font-size: 42px;">{total_kudos} kudos</p>'
    st.markdown(total_kudos, unsafe_allow_html=True)


####################
# TOP PERFORMANCES #
####################

longest_distance = processed_data[(processed_data.year == year) & (processed_data.type.isin(selected_activities))].distance.max()
longest_ride = processed_data[(processed_data.year == year) & (processed_data.type.isin(selected_activities))].moving_time.max()
most_elevation = processed_data[(processed_data.year == year) & (processed_data.type.isin(selected_activities))].total_elevation_gain.max()
fastest_pace = processed_data[(processed_data.year == year) & (processed_data.type.isin(selected_activities))].average_speed.max()

# Writing out
st.markdown("""---""")
# top_performance_title =  f'<H2 style="font-family:sans-serif; font-size: 24px;">Active Days</H2>'
# st.markdown(top_performance_title, unsafe_allow_html=True)
# performance_layout = 
    
    
#################
# CALENDAR VIEW #
#################

grouped_by_day_and_type = processed_data.groupby(['start_date_local', 'type']).agg({'id': 'count'}).reset_index()
# grouped_by_day = processed_data.groupby('start_date_local').agg({'id': 'count'}).reset_index()
# Converting to datetime
grouped_by_day_and_type.start_date_local = pd.to_datetime(grouped_by_day_and_type.start_date_local)
# # Set activity date as index
grouped_by_day_and_type.set_index('start_date_local', inplace=True)
# Drop duplicate index
grouped_by_day_and_type = grouped_by_day_and_type[~grouped_by_day_and_type.index.duplicated(keep='first')]
# Reindexing to fill in missing dates
# grouped_by_day_and_type = grouped_by_day_and_type.reindex(pd.date_range(start=grouped_by_day_and_type.index.min(), end=grouped_by_day_and_type.index.max()), fill_value=np.nan)
# Replacing 0 with na --days with multiple activities will be counted as 1
# grouped_by_day_and_type.id = grouped_by_day_and_type.id.replace(0, np.nan)
grouped_by_day_and_type.id = grouped_by_day_and_type.id.notnull().astype('int')

# Finding longest consecutive days with a workout
grouped_by_day_and_type = grouped_by_day_and_type[(grouped_by_day_and_type.index.year == year) & (grouped_by_day_and_type.type.isin(selected_activities))]
grouped_by_day_and_type = grouped_by_day_and_type.reset_index()
grouped_by_day_and_type = grouped_by_day_and_type.rename(columns={'index': 'start_date_local'})
grouped_by_day_and_type = grouped_by_day_and_type.sort_values(by='start_date_local', ascending=False)

# Summing the ids where dates are consecutive

grouped_by_day_and_type['days_between_activities'] = grouped_by_day_and_type.start_date_local.diff(-1).dt.days
grouped_by_day_and_type['is_consecutive'] = grouped_by_day_and_type.days_between_activities.eq(1)
grouped_by_day_and_type['streak'] = grouped_by_day_and_type.is_consecutive.groupby((grouped_by_day_and_type.is_consecutive != grouped_by_day_and_type.is_consecutive.shift()).cumsum()).cumsum()
grouped_by_day_and_type.set_index('start_date_local', inplace=True)

longest_streak = grouped_by_day_and_type.streak.max() + 1

active_days_title =  f'<h2 style="color:#45738F">Active Days</h2>'
st.markdown(active_days_title, unsafe_allow_html=True)

# Create a calplot calendar with the number of activities per day
# Grouping by day and activity type and counting the number of activities
try:
    # Create a calplot calendar with the number of activities per day for the selected year and activity type
    grouped_by_day_and_type = grouped_by_day_and_type[(grouped_by_day_and_type.index.year == year) & (grouped_by_day_and_type.type.isin(selected_activities))]
    fig, ax = calplot.calplot(data=grouped_by_day_and_type.id, colorbar=False, dropzero=True, edgecolor='grey', linewidth=0.5, cmap='tab20c', textcolor = '#808080')
    st.pyplot(fig=fig)
    st.markdown(f'<p style="font-family:sans-serif; font-size: 20px;"><i>Longest streak:</i> <b>{longest_streak}<b> days</p>', unsafe_allow_html=True)
except:
    st.write('No data available')

st.markdown("""---""")

top_performance_title =  f'<h2 style="color:#45738F">Top Performances</h2>'
st.markdown(top_performance_title, unsafe_allow_html=True)

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

try:
    # Longest ride
    longest_ride = processed_data[(processed_data.year == year) & (processed_data.type.isin(selected_activities))].distance.max()
    # Fethcing the activity name that matches the longest ride
    longest_activity_name = processed_data[(processed_data.year == year) & (processed_data.type.isin(selected_activities)) & (processed_data.distance == longest_ride)].name.values[0]
    # Fetching the activity date that matches the longest ride
    longest_activity_date = processed_data[(processed_data.year == year) & (processed_data.type.isin(selected_activities)) & (processed_data.distance == longest_ride)].start_date_local.values[0]
    longest_ride = round(longest_ride, 1)

    # Most elevation
    most_elevation = processed_data[(processed_data.year == year) & (processed_data.type.isin(selected_activities))].total_elevation_gain.max()

    # Fethcing the activity name that matches the most elevation
    most_elevation_name = processed_data[(processed_data.year == year) & (processed_data.type.isin(selected_activities)) & (processed_data.total_elevation_gain == most_elevation)].name.values[0]
    # Fetching the activity date that matches the most elevation
    most_elevation_date = processed_data[(processed_data.year == year) & (processed_data.type.isin(selected_activities)) & (processed_data.total_elevation_gain == most_elevation)].start_date_local.values[0]
    most_elevation = round(most_elevation, 1).astype(int)

    # Fastest pace
    fastest_ride = processed_data[(processed_data.year == year) & (processed_data.type.isin(selected_activities))].average_speed.max()
    # Fethcing the activity name that matches the fastest pace
    fastest_ride_name = processed_data[(processed_data.year == year) & (processed_data.type.isin(selected_activities)) & (processed_data.average_speed == fastest_ride)].name.values[0]
    # Fetching the activity date that matches the fastest pace
    fastest_ride_date = processed_data[(processed_data.year == year) & (processed_data.type.isin(selected_activities)) & (processed_data.average_speed == fastest_ride)].start_date_local.values[0]
    fastest_pace = round(fastest_ride, 1)

    # Longest day on the bike
    longest_day = processed_data[(processed_data.year == year) & (processed_data.type.isin(selected_activities))].moving_time.max()
    # Fetching the activity name that matches the longest day on the bike
    longest_day_name = processed_data[(processed_data.year == year) & (processed_data.type.isin(selected_activities)) & (processed_data.moving_time == longest_day)].name.values[0]
    # Fetching the activity date that matches the longest day on the bike
    longest_day_date = processed_data[(processed_data.year == year) & (processed_data.type.isin(selected_activities)) & (processed_data.moving_time == longest_day)].start_date_local.values[0]
    longest_day = round(longest_day, 1)

    # Best power
    best_power = processed_data[(processed_data.year == year) & (processed_data.type.isin(selected_activities))].average_watts.max()
    # Fetching the activity name that matches the best power
    best_power_name = processed_data[(processed_data.year == year) & (processed_data.type.isin(selected_activities)) & (processed_data.average_watts == best_power)].name.values[0]
    # Fetching the activity date that matches the best power
    best_power_date = processed_data[(processed_data.year == year) & (processed_data.type.isin(selected_activities)) & (processed_data.average_watts == best_power)].start_date_local.values[0]
    best_power = round(best_power, 1).astype(int)


    with kpi1:
        st.image('./icons/distance_2.png', width=50)
        st.markdown(f'<H3 style="font-family:sans-serif; font-size: 20px;">Longest ride: {longest_ride:,} miles</H3>', unsafe_allow_html=True)
        st.write(longest_activity_name)
        st.write(longest_activity_date)
        
    with kpi2:
        st.image('./icons/goal.png', width=50)
        st.markdown(f'<H3 style="font-family:sans-serif; font-size: 20px;">Most elevation: {most_elevation:,} feet</H3>', unsafe_allow_html=True)
        st.write(most_elevation_name)
        st.write(most_elevation_date)
        
    with kpi3:
        st.image('./icons/rocket.png', width=50)
        st.markdown(f'<H3 style="font-family:sans-serif; font-size: 20px;">Fastest pace: {fastest_ride} mph</H3>', unsafe_allow_html=True)
        st.write(fastest_ride_name)
        st.write(fastest_ride_date)
        
    with kpi4:
        st.image('./icons/clock.png', width=50)
        st.markdown(f'<H3 style="font-family:sans-serif; font-size: 20px;">Longest day: {longest_day} hours</H3>', unsafe_allow_html=True)
        st.write(longest_day_name)
        st.write(longest_day_date)
        
    with kpi5:
        st.image('./icons/power.png', width=50)
        st.markdown(f'<H3 style="font-family:sans-serif; font-size: 20px;">Best power: {best_power} watts</H3>', unsafe_allow_html=True)
        st.write(best_power_name)
        st.write(best_power_date)

except:
    st.write('No data available')
    




# st.dataframe(processed_data.head(5))