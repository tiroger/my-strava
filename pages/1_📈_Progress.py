#############
# LIBRARIES #
#############

from get_strava_data import my_data, process_data, bike_data, get_elev_data_GOOGLE, fetch_activity_streams # Functions to retrive data using strava api and process for visualizations

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

import os

###############
# CREDENTIALS #
###############

# token = MAPBOX_TOKEN = st.secrets['MAPBOX_TOKEN']
# GOOGLE_API_KEY = st.secrets['GOOGLE_API_KEY']

CLIENT_ID = os.environ['CLIENT_ID']
CLIENT_SECRET = os.environ['CLIENT_SECRET']
REFRESH_TOKEN = os.environ['REFRESH_TOKEN']
GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']

MAPBOX_TOKEN = os.environ['MAPBOX_TOKEN']


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

current_year = dt.datetime.today().year


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

# @st.cache_data(show_spinner=False, max_entries=5, ttl=86400)
# def fetch_activities():
#     with st.spinner('Data Refreshing...'):

#         my_data_df = my_data()
#         processed_data = process_data(my_data_df)

#         return processed_data

# processed_data = fetch_activities()



# Get local data # For development
processed_data = pd.read_csv('./data/processed_data.csv')
# bikes_df = pd.read_csv('./data/bike_data.csv')
athlete_df = pd.read_csv('./data/athlete_data.csv')



processed_data['start_date_local'] = pd.to_datetime(processed_data['start_date_local'])
processed_data['start_date_local'] = processed_data['start_date_local'].dt.strftime('%m-%d-%Y')

# Saving processed data to csv for future use
# processed_data.to_csv('./data/processed_data.csv', index=False)


processed_data['activity_date'] = pd.to_datetime(processed_data['start_date_local'])
# last_8_weeks_data = processed_data[processed_data.activity_date >= (dt.datetime.today() - pd.DateOffset(weeks=8))]
# exclude_run = last_8_weeks_data[last_8_weeks_data['type'] != 'Run']
# activity_ids_for_last_8_weeks = exclude_run['id'].tolist()


# @st.cache_data(show_spinner=False, max_entries=5, ttl=86400)
# def get_last_8_weeks_data():
#     data_dict = {}
#     for activity_id in activity_ids_for_last_8_weeks:
#         stream = fetch_activity_streams(activity_id)
#         data_dict[activity_id] = stream

#     df_list = []
#     for id_key, data in data_dict.items():
#         temp_df = pd.DataFrame(data)
#         temp_df['Id'] = id_key
#         df_list.append(temp_df)

#     # Combine all individual DataFrames into one
#     combined_df = pd.concat(df_list, ignore_index=True)

#     return combined_df

# last_8_weeks_data = get_last_8_weeks_data()

# windows = [1,2,5,10,15,20,30,45,60,90,120,180,240,300,360,420,480,540,600,660,720,780,840,900,960,1020,1080,1140,1200,1260,1320,1380,1440, 1500, 1560, 1620, 1680, 1740, 1800, 1860, 1920, 1980, 2040, 2100, 2160, 2220, 2280, 2340, 2400, 2460, 2520, 2580, 2640, 2700, 2760, 2820, 2880, 2940, 3000, 3060, 3120, 3180, 3240, 3300, 3360, 3420, 3480, 3540, 3600, 3660, 3720, 3780, 3840, 3900, 3960, 4020, 4080, 4140, 4200, 4260, 4320, 4380, 4440, 4500, 4560, 4620, 4680, 4740, 4800, 4860, 4920, 4980, 5040, 5100, 5160, 5220, 5280, 5340, 5400, 5460, 5520, 5580, 5640, 5700, 5760, 5820, 5880, 5940, 6000, 6060, 6120, 6180, 6240, 6300, 6360, 6420, 6480, 6540, 6600, 6660, 6720, 6780, 6840, 6900, 6960, 7020, 7080, 7140, 7200, 7260, 7320, 7380, 7440, 7500, 7560, 7620, 7680, 7740, 7800, 7860, 7920, 7980, 8040, 8100, 8160, 8220, 8280, 8340, 8400, 8460, 8520, 8580, 8640, 8700, 8760, 8820, 8880, 8940]

# best_rolling = {}
# for window in windows:
#     rolling = last_8_weeks_data.groupby('Id')['watts'].rolling(window=window, min_periods=1).mean()
#     best_rolling[window] = rolling.max()

# best_rolling_df = pd.DataFrame(list(best_rolling.items()), columns=['Duration', 'Value'])


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

st.markdown('<h2 style="color:#45738F">Yearly Progressions and Goals</h2>', unsafe_allow_html=True)

# activity_type = st.selectbox('Filter by sport', ['Ride','VirtualRide', 'Run']) # Select from dropdown
with st.sidebar:
    activity_type = st.multiselect('Select Activity Type', ['Ride','VirtualRide', 'Run'], default=['Ride', 'VirtualRide']) # Select from checkbox

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
grouped_by_year_and_month['date'] = pd.to_datetime(grouped_by_year_and_month['date'])  #f'{current_year}'

grouped_by_year_and_month = grouped_by_year_and_month.set_index('date').reindex(pd.date_range(start='2012-01-01', end=f'{current_year}-12-31'), fill_value=0).reset_index().rename(columns={'index': 'date'})
grouped_by_year_and_month['year'] = grouped_by_year_and_month['date'].dt.year
grouped_by_year_and_month['month'] = grouped_by_year_and_month['date'].dt.month
grouped_by_year_and_month['day'] = grouped_by_year_and_month['date'].dt.day

grouped_by_year_and_month['Cumulative Distance'] = grouped_by_year_and_month.groupby(['year'])['distance'].cumsum()
grouped_by_year_and_month['Cumulative Elevation'] = grouped_by_year_and_month.groupby(['year'])['total_elevation_gain'].cumsum()

# Adding day counter for each year
grouped_by_year_and_month['day_counter'] = grouped_by_year_and_month.groupby(['year']).cumcount() + 1

# Removing upcoming all dates after today
grouped_by_year_and_month = grouped_by_year_and_month[grouped_by_year_and_month.date <= dt.datetime.today()]


# Getting current year
current_year = dt.datetime.today().year

# Plotly charts
with st.sidebar:
    # try:
    #     selected_year = st.multiselect('Select year', grouped_by_year_and_month.year.unique(), default=[date for date in range(2012, current_year+1)]) # Filter for year
    # except: # If no data is available, we'll just show the current year
    #     st.warning('No data available for the selected year')
    #     selected_year = [this_year]
    selected_metric = st.selectbox('Metric', ['Cumulative Distance', 'Cumulative Elevation']) # Filter for desired metric

best_distance = grouped_by_year_and_month['Cumulative Distance'].max()
best_distance_year = grouped_by_year_and_month[grouped_by_year_and_month['Cumulative Distance'] == best_distance]['year'].unique()

best_elevation = grouped_by_year_and_month['Cumulative Elevation'].max()
best_elevation_year = grouped_by_year_and_month[grouped_by_year_and_month['Cumulative Elevation'] == best_elevation]['year'].unique()

selected_year = grouped_by_year_and_month.year.unique()

# Filtering year and activity type
# grouped_by_year_and_month = grouped_by_year_and_month[grouped_by_year_and_month['type'].isin([activity_type])]
grouped_by_year_and_month = grouped_by_year_and_month[grouped_by_year_and_month['year'].isin(selected_year)]


# Fetching the day counter for today's date
today_date = dt.datetime.today().strftime('%Y-%m-%d')
# Fetching the day counter for yesterday's date
today_counter = grouped_by_year_and_month[grouped_by_year_and_month.date == today_date].day_counter



# Projections for current year
current_year = dt.datetime.today().year
current_year_df = grouped_by_year_and_month[grouped_by_year_and_month['year'] == current_year]
current_year_avg_daily_distance = current_year_df['distance'].sum() / current_year_df['day_counter'].max()
current_year_avg_daily_elevation = current_year_df['total_elevation_gain'].sum() / current_year_df['day_counter'].max()

on_pace_distance_current_year = current_year_df['Cumulative Distance'].max() + current_year_avg_daily_distance * (365 - current_year_df['day_counter'].max())
on_pace_elevation_current_year = current_year_df['Cumulative Elevation'].max() + current_year_avg_daily_elevation * (365 - current_year_df['day_counter'].max())

# Dataframe with each theorethical day's distance and elevation thru end of year based on current pace
pace = {
    'day_counter': [i for i in range(1, 366)],
    # Distance per day is the current daily average distance
    'distance': [current_year_avg_daily_distance for i in range(1, 366)],
    # Elevation per day is the current daily average elevation
    'total_elevation_gain': [current_year_avg_daily_elevation for i in range(1, 366)]
}

pace_df = pd.DataFrame(pace)
pace_df['Cumulative Distance'] = pace_df['distance'].cumsum()
pace_df['Cumulative Elevation'] = pace_df['total_elevation_gain'].cumsum()

# daily_distance_2023 = grouped_by_year_and_month[grouped_by_year_and_month.year == f'{current_year}']['distance'].sum() / grouped_by_year_and_month[grouped_by_year_and_month.year == f'{current_year}']['day_counter'].max()
# on_pace_for_2023_distance = grouped_by_year_and_month[grouped_by_year_and_month.year == f'{current_year}']['Cumulative Distance'].max() + daily_distance_2023 * (365 - grouped_by_year_and_month[grouped_by_year_and_month.year == f'{current_year}']['day_counter'].max())



daily_elevation_2023 = grouped_by_year_and_month[grouped_by_year_and_month.year == f'{current_year}']['total_elevation_gain'].sum() / grouped_by_year_and_month[grouped_by_year_and_month.year == f'{current_year}']['day_counter'].max()
on_pace_for_2023_elevation = grouped_by_year_and_month[grouped_by_year_and_month.year == f'{current_year}']['Cumulative Elevation'].max() + daily_elevation_2023 * (365 - grouped_by_year_and_month[grouped_by_year_and_month.year == f'{current_year}']['day_counter'].max())

# Today's total distance
today_distance = grouped_by_year_and_month[grouped_by_year_and_month.date == today_date]['Cumulative Distance'].sum()

# Today's total elevation
today_elevation = grouped_by_year_and_month[grouped_by_year_and_month.date == today_date]['Cumulative Elevation'].sum()

# Plotly charts
plotly_chart_col, spacer_col, distance_col, elevation_col = st.columns([2, 0.1, 1, 1])

with plotly_chart_col:
    fig = px.line(grouped_by_year_and_month, x='day_counter', y=selected_metric, color='year')
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    current_month = dt.datetime.today().month
    # Highlight the current month using HTML bold tag
    ticktexts = ticktexts = [
    f'<span style="color: #FF4500;"><b>{month}</b></span>' if index + 1 == current_month else month
    for index, month in enumerate(months)
]
    fig.update_layout(
            xaxis=dict(
                showline=True,
                showgrid=True,
                showticklabels=True,
                linecolor='rgb(204, 204, 204)',
                linewidth=1,
                ticks='outside',
                tickvals=[1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335],
                ticktext=ticktexts,
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
                autorange=True,
                # format=',.0f',
            ),
            autosize=True,
            hovermode="x unified",
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title='',
            yaxis_title='Distance (mi)' if selected_metric == 'Cumulative Distance' else 'Elevation (ft)',
            margin=dict(l=0, r=0, t=0, b=0),
            yaxis_tickformat = ',.0f',
        )
    current_year_data = grouped_by_year_and_month[grouped_by_year_and_month['year'] == current_year]
    current_year_by_month_totals = current_year_data.groupby('month').agg({'distance': 'sum', 'total_elevation_gain': 'sum'}).reset_index()

    # Adding a vertical line for today's date
    fig.add_vline(
        x=today_counter.iloc[0], 
        line_width=0.5, 
        line_dash="dash", 
        line_color="grey")

    # Adding annotations for projected distance
    fig.add_annotation(
        xref="x",
        yref="paper",
        x=today_counter.iloc[0],
        y=0.75,
        text=f'On pace for {on_pace_distance_current_year.astype(int):,} mi' if selected_metric == 'Cumulative Distance' else f'On pace for {on_pace_elevation_current_year.astype(int):,} ft',
        showarrow=False,
        arrowcolor='#41484A',
        font=dict(size=18, color='#26A39E'),
        align="center",
        ax=0,
        ay=-50,
        # bordercolor="#41484A",
        # borderwidth=2,
        # borderpad=4,
        # bgcolor="#FF5622",
        # opacity=0.8
    )

    # fig.for_each_trace(lambda trace: fig.add_annotation(
    #     x=trace.x[-1], y=trace.y[-1], text='  '+trace.name, 
    #     font=dict(size=16, color=trace.line.color),
    #     ax=10, ay=10, xanchor="left", showarrow=False))
    fig.update_traces(mode="lines", hovertemplate=None)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    
    for trace in fig.data:
        if trace.name == str(current_year):  # Convert current_year to string for comparison
            trace.line.color = '#FF4500'  # or any color you want for the current year
        else:
            trace.line.color = '#949494'  # Making other years grey
    # for trace in fig.data:
    #     if trace.name != str(current_year):  # Assuming 'year' is a string. Adjust if it's int.
    #         trace.line.color = 'grey'  # Set non-current year lines to grey

    # Update annotations to reflect the color change
    for trace in fig.data:
        if trace.name == str(current_year):  # Check if the trace represents the current year
            trace.line.color = trace.line.color  # Keep the original color for the current year
            trace.line.width = 5  # Make line thicker for the current year
        else:
            trace.line.color = '#949494'  # Change color to grey for other years
            trace.line.width = 1  # Make other lines thinner
        font_size = [16 if trace.name == str(current_year) else 12][0]  # Set font size based on the condition
        fig.add_annotation(
            x=trace.x[-1], y=trace.y[-1], text='  '+trace.name, 
            font=dict(size=font_size,
                      color =[trace.line.color if trace.name == str(current_year) else '#949494'][0],
                      ),  # Use the determined color here
            ax=10, ay=10, xanchor="left", showarrow=False
    )

    average_month_width = 30.44  # Average days in a month
    tickvals = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    month_widths = [t2 - t1 - 1 for t1, t2 in zip(tickvals[:-1], tickvals[1:])]  # Subtract 1 to reduce width slightly

    # Calculate midpoints for each month to use as x values for bars
    # This makes each bar centered on its respective month
    month_midpoints = [(t1 + t2) / 2 for t1, t2 in zip(tickvals[:-1], tickvals[1:])]

    # Update your bar trace
    # Note: We're mapping the 'month' value to the correct midpoint based on your data structure
    fig.add_trace(go.Bar(
        x=[month_midpoints[month - 1] for month in current_year_by_month_totals['month']],  # Align 'month' with midpoints
        
        # monthly_metric = 'distance' if selected_metric == 'Cumulative Distance' else 'total_elevation_gain',
        y=current_year_by_month_totals['distance' if selected_metric == 'Cumulative Distance' else 'total_elevation_gain'],
        marker_color='#FC4F31',
        name='Monthly Total',
        opacity=0.25,
        width=month_widths,
        showlegend=True,
        hoverinfo='skip',
        hovertemplate='%{y:,.0f}mi' if selected_metric == 'Cumulative Distance' else '%{y:,.0f}ft',
        
    ))

    st.plotly_chart(fig, use_container_width=True, config= dict(
                displayModeBar = False,
                responsive = False
                ))


##################
# Bests and Goals #
##################

# col1, col2 = st.columns(2)

# with col1:
#     st.metric(f'Most Miles in a Year achieved in {best_distance_year[0]}', "{:,}".format(round(best_distance, 0).astype(int)) + ' miles')
# with col2:
#     st.metric(f'Most Elevation Gain in a Year achieved in {best_elevation_year[0]}', "{:,}".format(round(best_elevation, 0).astype(int)) + ' feet')

previous_year = this_year - 1
# Getting previous year's best distance
previous_best_distance = grouped_by_year_and_month[grouped_by_year_and_month['year'] == previous_year]['Cumulative Distance'].max()
# Getting previous year's best elevation
previous_best_elevation = grouped_by_year_and_month[grouped_by_year_and_month['year'] == previous_year]['Cumulative Elevation'].max()

# st.markdown("""---""")

# Limiting the data to today's date
# First day of the current year
d0 = dt.datetime(this_year, 1, 1)
# d0 = dt.datetime(2022, 1, 1)
d1 = dt.datetime.today()
delta = d1 - d0

yoy_improvement_percentage = 1.1

days_gone_by = delta.days # number of days since the beginning of the year
with st.sidebar:
    value_distance_goal = int(previous_best_distance.astype(int) * yoy_improvement_percentage)
    distance_goal = st.number_input("Choose a distance goal for the year", value=value_distance_goal) # distance goal
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

# col1, col2, = st.columns(2)

# with col1:
#     st.metric(f'{today_year} Distance Goal', "{:,}".format(distance_goal) + ' miles')
# with col2:
#     st.metric(f'Distance through {today.strftime("%m/%d/%Y")}', "{:,}".format(round(today_distance, 1)) + ' miles', f'{pace} ' + 'miles behind' if pace <0 else f'{pace} ' + 'miles ahead')

with st.sidebar:
    value_elev_goal = int(previous_best_elevation.astype(int) * yoy_improvement_percentage)
    elevation_goal = st.number_input("Choose an elevation goal for the year", value=value_elev_goal) # distance goal
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

# col1, col2, = st.columns(2)

# with col1:
#     st.metric(f'{today_year} Elevation Goal', "{:,}".format(elevation_goal) + ' feet')
# with col2:
#     st.metric(f'Elevation Gain through {today.strftime("%m/%d/%Y")}', "{:,}".format(round(today_elevation, 1)) + ' feet', f'{"{:,}".format(pace_elev)} ' + 'feet behind' if pace_elev <0 else f'{"{:,}".format(pace_elev)} ' + 'feet ahead')
    
# Other metrics -- calories, time, etc
aggreated_data = processed_data.groupby(['year', 'month', 'day']).agg({'distance': 'sum', 'total_elevation_gain': 'sum', 'moving_time': 'sum', 'kilojoules': 'sum'}).reset_index()

# st.markdown("""---""")

with distance_col:
    st.metric(f'Most Miles in a Year ({best_distance_year[0]})', "{:,}".format(round(best_distance, 0).astype(int)) + ' mi')
    st.metric(f'{today_year} Distance Goal', "{:,}".format(distance_goal) + ' mi')
    st.metric(f'Distance through {today.strftime("%m.%d.%Y")}', "{:,}".format(round(today_distance, 1)) + ' mi', f'{pace} ' + 'mi behind' if pace <0 else f'{pace} ' + 'miles ahead')
    

with elevation_col:
    st.metric(f'Most Elevation Gain in a Year ({best_elevation_year[0]})', "{:,}".format(round(best_elevation, 0).astype(int)) + ' ft')
    st.metric(f'{today_year} Elevation Goal', "{:,}".format(elevation_goal) + ' ft')
    st.metric(f'Elevation Gain through {today.strftime("%m.%d.%Y")}', "{:,.0f}".format(round(today_elevation, 1)) + ' ft', f'{"{:,.0f}".format(pace_elev)} ' + 'ft behind' if pace_elev <0 else f'{"{:,}".format(pace_elev)} ' + 'ft ahead')
    
    
current_distance = today_distance
curren_elevation_gain = today_elevation
distance_goal = distance_goal
elevation_goal = elevation_goal


# Create horizontal bar chart
fig = go.Figure()

# Bar for the accumulated distance
fig.add_trace(go.Bar(
    x=[current_distance],
    y=[''],  # Empty string for y-axis label
    orientation='h',
    name='Current Distance',
    marker_color='#FF4500',  # Light green color, adjust as needed
    width=0.6,  # Adjust for bar thickness
    hovertext='Current Distance: %{x} miles',
    hovertemplate='Current Distance: %{x} miles',
    hoverlabel=dict(
        bgcolor='rgba(76, 175, 80, 0.6)',
        font_size=16
    ),
    text = f'{current_distance:,.0f} mi',
    textposition='inside',
    insidetextanchor='start',
    insidetextfont=dict(
        size=16,
        color='white'
    )
))

# black line ar today's total distance
fig.add_vline(
    x=current_distance, 
    line_width=5, 
    line_color="black"
    )

fig.add_annotation(
    xref="x",
    yref="paper",
    x=current_distance*0.45,
    y=1,
    text='TODAY',
    showarrow=False,
    font=dict(size=10, color='#26A39E'),
)

# Bar for the remaining distance to reach the goal (if applicable)
if current_distance < distance_goal:
    fig.add_trace(go.Bar(
        x=[distance_goal - current_distance],
        y=[''],  # Empty string for y-axis label
        orientation='h',
        name='Remaining Distance',
        marker_color='#A9A9A9',  # Light red color, adjust as needed
        width=0.6,  # Adjust for bar thickness
        hovertext='Remaining Distance: %{x} miles',
        hovertemplate='Remaining Distance: %{x} miles',
        hoverlabel=dict(
            bgcolor='rgba(255, 87, 34, 0.6)',
            font_size=16
        ),
        # text = f'{(distance_goal - current_distance):,.0f} miles',
        # textposition='inside',
        # insidetextanchor='start',
        # insidetextfont=dict(
        #     size=16,
        #     color='white'
        # )
    ))


# Update the layout to hide all axes and place the legend at the bottom
fig.update_layout(
    xaxis=dict(
        showticklabels=False,  # Hide x-axis tick labels
        showgrid=False,  # Hide x-axis grid lines
        zeroline=False,  # Hide the zero line
        showline=False  # Hide the axis line
    ),
    yaxis=dict(
        showticklabels=False,  # Hide y-axis tick labels
        showgrid=False,  # Hide y-axis grid lines
        zeroline=False,  # Hide the zero line
        showline=False  # Hide the axis line
    ),
    barmode='stack',  # Stack the 'current' and 'remaining' bars
    showlegend=True,  # Show the legend
    legend=dict(
        orientation="h",  # Horizontal orientation of the legend
        yanchor="bottom",
        y=-0.25,  # Position of the legend (adjust as necessary)
        xanchor="center",
        x=0.5
    ),
    plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
    margin=dict(l=0, r=0, t=0, b=0)  # Reduce default margins
)
fig.update_layout(width=180, height=150)

with distance_col:
    st.plotly_chart(fig, use_container_width=True, config= dict(
                displayModeBar = False,
                responsive = False
                ))
    
    
fig = go.Figure()

# Bar for the accumulated distance
fig.add_trace(go.Bar(
    x=[curren_elevation_gain],
    y=[''],  # Empty string for y-axis label
    orientation='h',
    name='Current Elevation',
    marker_color='#FF4500',  # Light green color, adjust as needed
    width=0.6,  # Adjust for bar thickness
    hovertext='Current Elevation: %{x} feet',
    hovertemplate='Elevation: %{x} feet',
    hoverlabel=dict(
        bgcolor='rgba(76, 175, 80, 0.6)',
        font_size=16
    ),
    text = f'{curren_elevation_gain:,.0f} ft',
    textposition='inside',
    insidetextanchor='start',
    insidetextfont=dict(
        size=16,
        color='white'
    )
))

# black line at today's total distance
fig.add_vline(
    x=curren_elevation_gain, 
    line_width=5, 
    line_color="black")

fig.add_annotation(
    xref="x",
    yref="paper",
    x=curren_elevation_gain*0.2,
    y=1,
    text='TODAY',
    showarrow=False,
    font=dict(size=10, color='#26A39E'),
)

# Bar for the remaining distance to reach the goal (if applicable)
if curren_elevation_gain < elevation_goal:
    fig.add_trace(go.Bar(
        x=[elevation_goal - curren_elevation_gain],
        y=[''],  # Empty string for y-axis label
        orientation='h',
        name='Remaining Elevation',
        marker_color='#A9A9A9',  # Light red color, adjust as needed
        width=0.6,  # Adjust for bar thickness
        hovertext='Remaining Elevation: %{x} feet',
        hovertemplate='Elevation: %{x} feet',
        hoverlabel=dict(
            bgcolor='rgba(255, 87, 34, 0.6)',
            font_size=16
        ),
        # text = f'{(elevation_goal - curren_elevation_gain):,.0f} feet',
        # textposition='inside',
        # insidetextanchor='start',
        # insidetextfont=dict(
        #     size=16,
        #     color='white'
        # )
    ))

# Update the layout to hide all axes and place the legend at the bottom
fig.update_layout(
    xaxis=dict(
        showticklabels=False,  # Hide x-axis tick labels
        showgrid=False,  # Hide x-axis grid lines
        zeroline=False,  # Hide the zero line
        showline=False  # Hide the axis line
    ),
    yaxis=dict(
        showticklabels=False,  # Hide y-axis tick labels
        showgrid=False,  # Hide y-axis grid lines
        zeroline=False,  # Hide the zero line
        showline=False  # Hide the axis line
    ),
    barmode='stack',  # Stack the 'current' and 'remaining' bars
    showlegend=True,  # Show the legend
    legend=dict(
        orientation="h",  # Horizontal orientation of the legend
        yanchor="bottom",
        y=-0.25,  # Position of the legend (adjust as necessary)
        xanchor="center",
        x=0.5
    ),
    plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
    margin=dict(l=0, r=0, t=0, b=0)  # Reduce default margins
)
fig.update_layout(width=180, height=150)

with elevation_col:
    st.plotly_chart(fig, use_container_width=True, config= dict(
                displayModeBar = False,
                responsive = False
                ))
    

st.markdown("""---""")

###############
# POWER CURVE #
###############

offset_options = ['Last 4 weeks', 'Last 8 weeks', 'Last 12 weeks', 'Last 26 weeks', f'All of {current_year}']
week_offset_dropdown = st.selectbox('Select a time period for the power curve', (option for option in offset_options), index=1)

offset_options_dict = {
    'Last 4 weeks': 4,
    'Last 8 weeks': 8,
    'Last 12 weeks': 12,
    'Last 26 weeks': 26,
    f'All of {current_year}': 52
}

week_offset = offset_options_dict[week_offset_dropdown]
# st.write(week_offset)

# week_offset = 8
last_8_weeks_data = processed_data[processed_data.activity_date >= (dt.datetime.today() - pd.DateOffset(weeks=week_offset))]
# exclude_run = last_8_weeks_data[last_8_weeks_data['type'] != 'Run']
activity_ids_for_last_8_weeks = last_8_weeks_data['id'].tolist()

@st.cache_data(show_spinner=False, max_entries=5, ttl=86400)
def get_last_8_weeks_data():
    data_dict = {}
    for activity_id in activity_ids_for_last_8_weeks:
        stream = fetch_activity_streams(activity_id)
        data_dict[activity_id] = stream

    df_list = []
    for id_key, data in data_dict.items():
        temp_df = pd.DataFrame(data)
        temp_df['Id'] = id_key
        df_list.append(temp_df)

    # Combine all individual DataFrames into one
    combined_df = pd.concat(df_list, ignore_index=True)

    return combined_df

last_8_weeks_data = get_last_8_weeks_data()

@st.cache_data(show_spinner=False, max_entries=5, ttl=86400)
def plot_power_curve():
    windows = [1,2,5,10,15,20,30,45,60,90,120,180,240,300,360,420,480,540,600,660,720,780,840,900,960,1020,1080,1140,1200,1260,1320,1380,1440, 1500, 1560, 1620, 1680, 1740, 1800, 1860, 1920, 1980, 2040, 2100, 2160, 2220, 2280, 2340, 2400, 2460, 2520, 2580, 2640, 2700, 2760, 2820, 2880, 2940, 3000, 3060, 3120, 3180, 3240, 3300, 3360, 3420, 3480, 3540, 3600, 3660, 3720, 3780, 3840, 3900, 3960, 4020, 4080, 4140, 4200, 4260, 4320, 4380, 4440, 4500, 4560, 4620, 4680, 4740, 4800, 4860, 4920, 4980, 5040, 5100, 5160, 5220, 5280, 5340, 5400, 5460, 5520, 5580, 5640, 5700, 5760, 5820, 5880, 5940, 6000, 6060, 6120, 6180, 6240, 6300, 6360, 6420, 6480, 6540, 6600, 6660, 6720, 6780, 6840, 6900, 6960, 7020, 7080, 7140, 7200, 7260, 7320, 7380, 7440, 7500, 7560, 7620, 7680, 7740, 7800, 7860, 7920, 7980, 8040, 8100, 8160, 8220, 8280, 8340, 8400, 8460, 8520, 8580, 8640, 8700, 8760, 8820, 8880, 8940]

    best_rolling = {}
    for window in windows:
        rolling = last_8_weeks_data.groupby('Id')['watts'].rolling(window=window, min_periods=1).mean()
        best_rolling[window] = rolling.max()

    best_rolling_df = pd.DataFrame(list(best_rolling.items()), columns=['Duration', 'Value'])

    st.markdown(f'<h4 style="color:#45738F">Best Efforts Power Curve for the Past {week_offset} weeks</h4>', unsafe_allow_html=True)

    fig = px.line(best_rolling_df, x='Duration', y='Value', title='', labels={'Duration': '', 'Value': 'Power (watts)'},
                height=400
                )

    fig.update_layout(
        # remove backgrould and gridlines
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)'),
        yaxis=dict(showgrid=False),
        margin=dict(l=0, r=0, t=0, b=0),
           shapes=[
        # Line Horizontal y=0
        dict(
            type='line',
            yref='y', y0=0, y1=0,  # y-reference is set to 'y' to use plot's y-axis
            xref='paper', x0=0, x1=1,  # x-reference is set to 'paper' to span whole width
            line=dict(
                color="rgba(150, 150, 150, 0.5)",  # Set line color and transparency
                width=2,  # Set line width
                dash="dot",  # Set line style (optional, e.g., 'dash', 'dot', etc.)
            ),
        ),
    ]
        )
    fig.update_traces(line=dict(width=4))
    
    # Estimated FTP -- 95% of the best 20 minute power
    ftp = best_rolling_df[best_rolling_df['Duration'] == 1200]['Value'].max() * 0.95

    fig.add_hline(y=ftp, 
                  line_width=1, 
                  line_dash="dash", 
                  line_color="grey")

    x_labels = ['5s', '5min', '20min', '1hr', '2hr']
    x_labels = [f'<b>{label}</b>' for label in x_labels]
    fig.update_xaxes(
        tickvals=[5, 300, 600, 3600, 7200], 
        ticktext=x_labels)
    fig.add_annotation(
        x=8200,
        y=ftp+30,
        text=f'Estimated FTP: <b>{ftp:.0f} watts</b>',
        showarrow=False,
        arrowhead=1,
        font=dict(
            size=18,
            color='#FF4500'
        ),
        # bordercolor="black",
        # borderwidth=2,
        # borderpad=4,
        # bgcolor="white",
        # opacity=0.8
    )

    # Best 5 minute power
    best_5min = best_rolling_df[best_rolling_df['Duration'] == 300]['Value'].max()
    fig.add_annotation(
        x=350,
        y=600,
        text=f'Best 5min: <b>{best_5min:.0f}w</b>',
        showarrow=False,
        arrowhead=1,
        textangle=-90,
        font=dict(
            size=12,
            color='#4D4D4E'
        ),
        # bordercolor="black",
        # borderwidth=2,
        # borderpad=4,
        # bgcolor="white",
        # opacity=0.8
    )

    best_20min = best_rolling_df[best_rolling_df['Duration'] == 1200]['Value'].max()
    fig.add_annotation(
        x=650,
        y=600,
        text=f'Best 20min: <b>{best_20min:.0f}w</b>',
        showarrow=False,
        arrowhead=1,
        textangle=-90,
        font=dict(
            size=12,
            color='#4D4D4E'
        ),
        # bordercolor="black",
        # borderwidth=2,
        # borderpad=4,
        # bgcolor="white",
        # opacity=0.8
    )

    best_hour = best_rolling_df[best_rolling_df['Duration'] == 3600]['Value'].max()
    fig.add_annotation(
        x=3650,
        y=600,
        text=f'Best 1hr: <b>{best_hour:.0f}w</b>',
        showarrow=False,
        arrowhead=1,
        textangle=-90,
        font=dict(
            size=12,
            color='#4D4D4E'
        ),
        # bordercolor="black",
        # borderwidth=2,
        # borderpad=4,
        # bgcolor="white",
        # opacity=0.8
    )

    best_2hour = best_rolling_df[best_rolling_df['Duration'] == 7200]['Value'].max()
    fig.add_annotation(
        x=7250,
        y=600,
        text=f'Best 2hr: <b>{best_2hour:.0f}w</b>',
        showarrow=False,
        arrowhead=1,
        textangle=-90,
        font=dict(
            size=12,
            color='#4D4D4E'
        ),
        # bordercolor="black",
        # borderwidth=2,
        # borderpad=4,
        # bgcolor="white",
        # opacity=0.8
    )
    
    # # Horozontal line for best 15 second power
    # best_15s = best_rolling_df[best_rolling_df['Duration'] == 15]['Value'].max()
    # best_5min = best_rolling_df[best_rolling_df['Duration'] == 300]['Value'].max()
    
    # fig.add_annotation(
    #     x=8200,
    #     y=best_15s+30,
    #     text=f'Best 15s: <b>{best_15s:.0f}w</b>',
    #     showarrow=False,
    #     arrowhead=1,
    #     font=dict(
    #         size=12,
    #         color='#4D4D4E'
    #     ),
    #     # bordercolor="black",
    #     # borderwidth=2,
    #     # borderpad=4,
    #     # bgcolor="white",
    #     # opacity=0.8
    # )
    
    # fig.add_annotation(
    #     x=8200,
    #     y=best_5min+30,
    #     text=f'Best 5min: <b>{best_5min:.0f}w</b>',
    #     showarrow=False,
    #     arrowhead=1,
    #     font=dict(
    #         size=12,
    #         color='#4D4D4E'
    #     ),
    #     # bordercolor="black",
    #     # borderwidth=2,
    #     # borderpad=4,
    #     # bgcolor="white",
    #     # opacity=0.8
    # )
    
    # fig.add_hline(y=best_15s, 
    #               line_width=1, 
    #               line_dash="dash", 
    #               line_color="grey")
    
    # fig.add_annotation(
    #     x=100,
    #     y=best_15s+30,
    #     text=f'Best 15s: <b>{best_15s:.0f}w</b>',
    #     showarrow=False,
    #     arrowhead=1,
    #     font=dict(
    #         size=12,
    #         color='#4D4D4E'
    #     ),
    #     # bordercolor="black",
    #     # borderwidth=2,
    #     # borderpad=4,
    #     # bgcolor="white",
    #     # opacity=0.8
    
    # )

    st.plotly_chart(fig, use_container_width=True, config= dict(
                    displayModeBar = False,
                    responsive = False
                    ))

power_curve = plot_power_curve()