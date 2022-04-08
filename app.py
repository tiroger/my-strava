#############
# LIBRARIES #
#############

from turtle import color
from get_strava_data import my_data, process_data # Functions to retrive data using strava api and process for visualizations

import pandas as pd
import numpy as np
import datetime as dt

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


import streamlit as st
import streamlit.components.v1 as components

strava_color_palette = ['#FC4C02', '#45738F', '#DF553B', '#3A18B0', '#FFAA06', '#26A39E', '#951B05', '#D22B0C', '#F5674E'] # [strava orange, strava blue, warm orange, advance indigo, intermediate gold, beginner teal, hard red, medium sienna, easy peach]

# Strava widgets
latest_activities = 'https://www.strava.com/athletes/644338/latest-rides/53c89f1acdf2bf69a8bc937e3793e9bfb56d64be'
my_week = 'https://www.strava.com/athletes/644338/activity-summary/53c89f1acdf2bf69a8bc937e3793e9bfb56d64be'


#################################
# ACQUIRING AND PROCESSING DATA #
#################################


##### Use one of two options below #####

# # Get data using strava api # For deployment
# my_data_df = my_data()
# processed_data = process_data(my_data_df)

# Get local data # For development
processed_data = pd.read_csv('./data/processed_data.csv')


############
# SIDE BAR #
############

with st.sidebar:
    st.header('Latest Activities')
    components.iframe(my_week, height=170)
    components.iframe(latest_activities, height=500)


#############
# MAIN PAGE #
#############

st.title('MY JOURNEY ON STRAVA')

# Total number of activities
start_date = processed_data.year.min()
burger_calories = 354
total_activities = processed_data.id.count()
num_rides = (processed_data.type == 'Ride').sum()
num_runs = (processed_data.type == 'Workout').sum()
distance_traveled = processed_data.distance.sum().astype(int)
total_kudos = processed_data.kudos_count.sum()
earth_circumference = 24901 # earth circumference in miles 
perc_around_the_earth = (distance_traveled / earth_circumference)


# print(f'Strava user since: {start_date}')
# print(f'Total number of activities: {total_activities}')
# print(f'Total distance traveled: {"{:,}".format(distance_traveled)} miles or {"{:.0%}".format(perc_around_the_earth)} of the earth circumference')

# print(f'Number of Rides: {num_rides}')
# print(f'Number of Runs: {num_runs}')

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Activities", value=total_activities)
with col2:
    st.metric(label="Distance Traveled", value=f'{"{:,}".format(distance_traveled)} miles')
with col3:
    st.metric(label="Kudos", value="{:,}".format(total_kudos))

st.subheader('Activity Breakdown')

# Chart of all activities by type
breakdown_by_type = processed_data['type'].value_counts()

fig = px.bar(breakdown_by_type, x=breakdown_by_type.index, y=breakdown_by_type.values, text_auto='.0s') # Plotly Express
fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
fig.update_layout(
    # title_text="Activity Breakdown",
    yaxis_title="",
    xaxis_title="",
    font=dict(
        family="Arial",
        size=14,
    )
)
fig.update_traces(marker_color='#FC4C02',
                  marker_line_width=1.5, opacity=0.6)
                  
fig.update_yaxes(showticklabels=False)
st.plotly_chart(fig, use_container_width=True)

####################################
# BASIC ANALYSIS AND VISUALIZATION #
####################################

