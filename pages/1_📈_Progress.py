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

# from PIL import Image
import base64

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


#################################
# Yearly Progressions line chart #
#################################
# Filter by activity type

st.markdown('<h2 style="color:#45738F">Year Progressions and Goals</h2>', unsafe_allow_html=True)

# activity_type = st.selectbox('Filter by sport', ['Ride','VirtualRide', 'Run']) # Select from dropdown
activity_type = st.multiselect('Select activity', ['Ride','VirtualRide', 'Run'], default=['Ride', 'VirtualRide']) # Select from checkbox

processed_data = processed_data[processed_data.type.isin(activity_type)]

# grouped_by_year_and_month = processed_data.groupby(['year', 'month']).agg({'distance': 'sum', 'total_elevation_gain': 'sum'}).reset_index() # Group by year and month

# # Since not all months have data, we're creating entries for missing months and setting the distance and elevation gain to 0
# mux = pd.MultiIndex.from_product([grouped_by_year_and_month.year.unique(), range(1,13)], names=['year','month'])
# grouped_by_year_and_month = grouped_by_year_and_month.set_index(['year', 'month']).reindex(mux, fill_value=0).reset_index()
# grouped_by_year_and_month['Cumulative Distance'] = grouped_by_year_and_month.groupby(['year'])['distance'].cumsum()
# grouped_by_year_and_month['Cumulative Elevation'] = grouped_by_year_and_month.groupby(['year'])['total_elevation_gain'].cumsum()

# grouped_by_year_and_month['month'] = grouped_by_year_and_month['month'].apply(lambda x: months[x -1])

# # Limiting data to current month
# months_left = months[this_month:]

# # Filtering out months beyond current one
# no_data_yet = grouped_by_year_and_month[grouped_by_year_and_month.year == this_year]
# no_data_yet = no_data_yet[no_data_yet.month.isin(months_left)]

# # Removing upcoming months with no data from dataframe
# grouped_by_year_and_month = grouped_by_year_and_month[~grouped_by_year_and_month.isin(no_data_yet)]
# # Dropping na years
# grouped_by_year_and_month = grouped_by_year_and_month.dropna(subset=['year'])

# grouped_by_year_and_month['year'] = grouped_by_year_and_month['year'].astype(int)

grouped_by_year_and_month = processed_data.groupby(['year', 'month', 'day']).agg({'distance': 'sum', 'total_elevation_gain': 'sum'}).reset_index()

# Creating a new date column
grouped_by_year_and_month['date'] = pd.to_datetime(grouped_by_year_and_month[['year', 'month', 'day']])
# converting date column to datetime
grouped_by_year_and_month['date'] = pd.to_datetime(grouped_by_year_and_month['date'])

grouped_by_year_and_month = grouped_by_year_and_month.set_index('date').reindex(pd.date_range(start='2012-01-01', end='2023-12-31'), fill_value=0).reset_index().rename(columns={'index': 'date'})
grouped_by_year_and_month['year'] = grouped_by_year_and_month['date'].dt.year
grouped_by_year_and_month['month'] = grouped_by_year_and_month['date'].dt.month
grouped_by_year_and_month['day'] = grouped_by_year_and_month['date'].dt.day

grouped_by_year_and_month['Cumulative Distance'] = grouped_by_year_and_month.groupby(['year'])['distance'].cumsum()
grouped_by_year_and_month['Cumulative Elevation'] = grouped_by_year_and_month.groupby(['year'])['total_elevation_gain'].cumsum()

# Adding day counter for each year
grouped_by_year_and_month['day_counter'] = grouped_by_year_and_month.groupby(['year']).cumcount() + 1

# Removing upcoming all dates after today
grouped_by_year_and_month = grouped_by_year_and_month[grouped_by_year_and_month.date <= dt.datetime.today()]


# Plotly charts
try:
    selected_year = st.multiselect('Select year', grouped_by_year_and_month.year.unique(), default=[date for date in range(2012, 2024)]) # Filter for year
except: # If no data is available, we'll just show the current year
    st.warning('No data available for the selected year')
    selected_year = [this_year]
selected_metric = st.selectbox('Metric', ['Cumulative Distance', 'Cumulative Elevation']) # Filter for desired metric

best_distance = grouped_by_year_and_month['Cumulative Distance'].max()
best_distance_year = grouped_by_year_and_month[grouped_by_year_and_month['Cumulative Distance'] == best_distance]['year'].unique()

best_elevation = grouped_by_year_and_month['Cumulative Elevation'].max()
best_elevation_year = grouped_by_year_and_month[grouped_by_year_and_month['Cumulative Elevation'] == best_elevation]['year'].unique()

# Filtering year and activity type
# grouped_by_year_and_month = grouped_by_year_and_month[grouped_by_year_and_month['type'].isin([activity_type])]
grouped_by_year_and_month = grouped_by_year_and_month[grouped_by_year_and_month['year'].isin(selected_year)]


# Fetching the day counter for today's date
today_date = dt.datetime.today().strftime('%Y-%m-%d')
# Fetching the day counter for yesterday's date
today_counter = grouped_by_year_and_month[grouped_by_year_and_month.date == today_date].day_counter.values[0]


# Projections for 2023
daily_distance_2023 = grouped_by_year_and_month[grouped_by_year_and_month.year == 2023]['distance'].sum() / grouped_by_year_and_month[grouped_by_year_and_month.year == 2023]['day_counter'].max()
on_pace_for_2023_distance = grouped_by_year_and_month[grouped_by_year_and_month.year == 2023]['Cumulative Distance'].max() + daily_distance_2023 * (365 - grouped_by_year_and_month[grouped_by_year_and_month.year == 2023]['day_counter'].max())

daily_elevation_2023 = grouped_by_year_and_month[grouped_by_year_and_month.year == 2023]['total_elevation_gain'].sum() / grouped_by_year_and_month[grouped_by_year_and_month.year == 2023]['day_counter'].max()
on_pace_for_2023_elevation = grouped_by_year_and_month[grouped_by_year_and_month.year == 2023]['Cumulative Elevation'].max() + daily_elevation_2023 * (365 - grouped_by_year_and_month[grouped_by_year_and_month.year == 2023]['day_counter'].max())

# Today's total distance
today_distance = grouped_by_year_and_month[grouped_by_year_and_month.date == today_date]['Cumulative Distance'].sum()

# Today's total elevation
today_elevation = grouped_by_year_and_month[grouped_by_year_and_month.date == today_date]['Cumulative Elevation'].sum()

# Plotly charts
fig = px.line(grouped_by_year_and_month, x='day_counter', y=selected_metric, color='year')
fig.update_layout(
        xaxis=dict(
            showline=True,
            showgrid=True,
            showticklabels=True,
            linecolor='rgb(204, 204, 204)',
            linewidth=1,
            ticks='outside',
            tickvals=[1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335],
            ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            tickfont=dict(
                family='Arial',
                size=12,
                color='rgb(82, 82, 82)',
            ),
        ),
        yaxis=dict(
            showgrid=True,
            zeroline=False,
            showline=False,
            gridcolor = 'rgb(235, 236, 240)',
            showticklabels=True,
            title='',
            autorange=True
        ),
        autosize=True,
        hovermode="x unified",
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title='',
        yaxis_title='Distamce (miles)' if selected_metric == 'Cumulative Distance' else 'Elevation (feet)',
        margin=dict(l=0, r=0, t=0, b=0)
    )
# Addding annotations for projected distance for 2023
fig.add_annotation(x=today_counter, y=today_distance if selected_metric == 'Cumulative Distance' else today_elevation, text=f'On pace for : {on_pace_for_2023_distance.astype(int):,} miles' if selected_metric == 'Cumulative Distance' else f'On pace for : {on_pace_for_2023_elevation.astype(int):,} feet', showarrow=True, font=dict(size=24, color='#26A39E'))

fig.for_each_trace(lambda trace: fig.add_annotation(
    x=trace.x[-1], y=trace.y[-1], text='  '+trace.name, 
    font_color=trace.line.color,
    ax=10, ay=10, xanchor="left", showarrow=False))
fig.update_traces(mode="lines", hovertemplate=None)
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)

st.plotly_chart(fig, use_container_width=True, config= dict(
            displayModeBar = False))


##################
# Bests and Goals #
##################

col1, col2 = st.columns(2)

with col1:
    st.metric(f'Most Miles in a Year achieved in {best_distance_year[0]}', "{:,}".format(round(best_distance, 0).astype(int)) + ' miles')
with col2:
    st.metric(f'Most Elevation Gain in a Year achieved in {best_elevation_year[0]}', "{:,}".format(round(best_elevation, 0).astype(int)) + ' feet')

previous_year = this_year - 1
# Getting previous year's best distance
previous_best_distance = grouped_by_year_and_month[grouped_by_year_and_month['year'] == previous_year]['Cumulative Distance'].max()
# Getting previous year's best elevation
previous_best_elevation = grouped_by_year_and_month[grouped_by_year_and_month['year'] == previous_year]['Cumulative Elevation'].max()

st.markdown("""---""")

# Limiting the data to today's date
# First day of the current year
d0 = dt.datetime(this_year, 1, 1)
# d0 = dt.datetime(2022, 1, 1)
d1 = dt.datetime.today()
delta = d1 - d0

days_gone_by = delta.days # number of days since the beginning of the year

distance_goal = st.number_input("Choose a distance goal for the year", value=previous_best_distance.astype(int) + 1) # distance goal for 2023
# st.write('The current distance goal is ', distance_goal)

monthly_goal = distance_goal/12 # monthly distance to reach goal
daily_goals = distance_goal/365 # daily distance to reach goal

# Cumulative distance per day
grouped_by_day = processed_data.groupby(['year', 'month', 'day']).agg({'distance': 'sum'}).reset_index()
# Daily cumulative distance
grouped_by_day['Cummulative Distance'] = grouped_by_day.groupby(['year'])['distance'].cumsum()

should_be_reached = daily_goals*days_gone_by # distance that should have been reached by now to be on pace for the desired goal for the year

today_year = dt.datetime.today().year
# print(f"Today's month is the {this_month}th month and year is {today_year}")


where_i_am = grouped_by_day[(grouped_by_day.year == today_year) & (grouped_by_day.month == this_month)]['Cummulative Distance'].max()
# print(f"I should have reached {should_be_reached} miles. I've done {where_i_am} miles")

pace = round(today_distance - should_be_reached, 1)

col1, col2, = st.columns(2)

with col1:
    st.metric(f'{today_year} Distance Goal', "{:,}".format(distance_goal) + ' miles')
with col2:
    st.metric(f'Distance through {today.strftime("%m/%d/%Y")}', "{:,}".format(round(today_distance, 1)) + ' miles', f'{pace} ' + 'miles behind' if pace <0 else f'{pace} ' + 'miles ahead')


elevation_goal = st.number_input("Choose an elevation goal for the year", value=previous_best_elevation.astype(int) + 1) # distance goal for 2022
# st.write('The current distance goal is ', distance_goal)

monthly_goal_elev = elevation_goal/12
daily_goals_elev = elevation_goal/365

# Cumulative distance per day
grouped_by_day = processed_data.groupby(['year', 'month', 'day']).agg({'total_elevation_gain': 'sum'}).reset_index()
# Daily cumulative distance
grouped_by_day['Cummulative Elevation'] = grouped_by_day.groupby(['year'])['total_elevation_gain'].cumsum()

should_be_reached_elev = daily_goals_elev*days_gone_by # elevation that should have been reached by now to be on pace to reach the set goal for the year

today_year = dt.datetime.today().year
# print(f"Today's month is the {this_month}th month and year is {today_year}")


where_i_am_elev = grouped_by_day[(grouped_by_day.year == today_year) & (grouped_by_day.month == this_month)]['Cummulative Elevation'].max()
# print(f"I should have reached {should_be_reached} miles. I've done {where_i_am} miles")

pace_elev = round(where_i_am_elev - should_be_reached_elev, 1)

col1, col2, = st.columns(2)

with col1:
    st.metric(f'{today_year} Elevation Goal', "{:,}".format(elevation_goal) + ' feet')
with col2:
    st.metric(f'Elevation Gain through {today.strftime("%m/%d/%Y")}', "{:,}".format(round(today_elevation, 1)) + ' feet', f'{"{:,}".format(pace_elev)} ' + 'feet behind' if pace_elev <0 else f'{"{:,}".format(pace_elev)} ' + 'feet ahead')