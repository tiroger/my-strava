#############
# LIBRARIES #
#############

from turtle import color, width
from get_strava_data import my_data, process_data # Functions to retrive data using strava api and process for visualizations

import pandas as pd
import numpy as np
import datetime as dt

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go


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

# Get data using strava api # For deployment
# my_data_df = my_data()
# processed_data = process_data(my_data_df)

# Get local data # For development
processed_data = pd.read_csv('./data/processed_data.csv')
processed_data['start_date_local'] = pd.to_datetime(processed_data['start_date_local'])
processed_data['start_date_local'] = processed_data['start_date_local'].dt.strftime('%m-%d-%Y')

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


############
# SIDE BAR #
############

# with st.sidebar: # Option to show strava widgets
#     st.header('Overview')
    # components.iframe(my_week, height=170)
    # components.iframe(latest_activities, height=500)

############
# Overview #
############

with st.sidebar:
    st.title('Overview')
    st.subheader(f'Member since {start_date}')


    col1, col2 = st.columns(2)
    with col1:
        st.image('./icons/dumbbell.png', width=80, output_format='PNG')
    with col2:
        st.metric(label="Activities", value=total_activities)

    col1, col2 = st.columns(2)
    with col1:
        st.image('./icons/stopwatch.png', width=80, output_format='PNG')
    with col2:
        st.metric(label="Moving Time (hours)", value=f'{total_time}')

    col1, col2 = st.columns(2)
    with col1:
        st.image('./icons/road.png', width=80, output_format='PNG')
    with col2:
        st.metric(label="Distance (miles)", value=f'{"{:,}".format(distance_traveled)}')

    col1, col2 = st.columns(2)
    with col1:
        st.image('./icons/mountain.png', width=100, output_format='PNG')
    with col2:
        st.metric(label="Elevation Gain (ft)", value=f'{"{:,}".format(feet_climbed)}')
    
    col1, col2 = st.columns(2)
    with col1:
        st.image('./icons/like.png', width=80, output_format='PNG')
    with col2:
        st.metric(label="Kudos", value="{:,}".format(total_kudos))

    ########################
    # Activities Pie chart #
    ########################

    grouped_by_type = processed_data.groupby('type').agg({'type': 'count'}).rename(columns={'type': 'total'}).sort_values('total', ascending=False).reset_index()
    grouped_by_type.loc[grouped_by_type.total < 20, 'type'] = 'Other'
    pie_df = grouped_by_type.groupby('type').agg({'total': 'sum'}).rename(columns={'total': 'total'}).reset_index()

    activities = pie_df.type
    breakdown_by_type = pie_df.total

    fig = go.Figure(data=[go.Pie(labels=activities, values=breakdown_by_type, hole=.7)])
    fig.update_traces(textposition='outside', textinfo='label+value')
    fig.update_layout(showlegend=False, uniformtext_minsize=16, uniformtext_mode='hide', hovermode=False, paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=0, b=0), annotations=[dict(text='Activities', x=0.5, y=0.5, font_size=20, showarrow=False)])

    st.plotly_chart(fig, use_container_width=True)

############
# SIDE BAR #
############


#############
# MAIN PAGE #
#############

st.title('MY JOURNEY ON STRAVA')
st.subheader('Activities')

####################
# Activities Table #
####################

# Filter by activity type
activity_type = st.selectbox('Filter by activity type', processed_data.type.unique()) # Select from dropdown

# Processing data for table
streamlit_df = processed_data[['start_date_local', 'name', 'type', 'moving_time', 'distance', 'total_elevation_gain', 'average_speed', 'average_cadence', 'average_watts', 'average_heartrate', 'suffer_score']]
streamlit_df['start_date_local'] = pd.to_datetime(streamlit_df['start_date_local'])
streamlit_df['start_date_local'] = streamlit_df['start_date_local'].dt.strftime('%m-%d-%Y')
streamlit_df.rename(columns={'start_date_local': 'Date','name': 'Name', 'type': 'Type', 'moving_time': 'Moving Time (hours)', 'distance': 'Distance (miles)', 'total_elevation_gain': 'Elevation Gain (ft)', 'average_speed': 'Average Speed (mph)', 'average_cadence': 'Average Cadence (rpm)', 'average_watts': 'Average Watts', 'average_heartrate': 'Average Heartrate', 'suffer_score': 'Suffer Score'}, inplace=True)
streamlit_df.set_index('Date', inplace=True)
streamlit_df = streamlit_df[streamlit_df['Type'].isin([activity_type])]

st.dataframe(streamlit_df)

#################################
# Yearly Progression line chart #
#################################

st.subheader('Year Progressions')

grouped_by_year_and_month = processed_data.groupby(['year', 'month', 'type']).agg({'distance': 'sum', 'total_elevation_gain': 'sum'}).reset_index() # Group by year and month

# Since not all months have data, we're creating entries for missing months and setting the distance and elevation gain to 0
mux = pd.MultiIndex.from_product([grouped_by_year_and_month.year.unique(), grouped_by_year_and_month.type.unique(), range(1,13)], names=['year', 'type' ,'month'])
grouped_by_year_and_month = grouped_by_year_and_month.set_index(['year', 'type', 'month']).reindex(mux, fill_value=0).reset_index()
grouped_by_year_and_month['Cummulative Distance'] = grouped_by_year_and_month.groupby(['year', 'type'])['distance'].cumsum()
grouped_by_year_and_month['Cummulative Elevation'] = grouped_by_year_and_month.groupby(['year', 'type'])['total_elevation_gain'].cumsum()
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
grouped_by_year_and_month['month'] = grouped_by_year_and_month['month'].apply(lambda x: months[x -1])

# Plotly charts

selected_year = st.multiselect('Filter by Year', grouped_by_year_and_month.year.unique(), default=grouped_by_year_and_month.year.max()) # Filter for year
selected_metric = st.selectbox('Metric', ['Cummulative Distance', 'Cummulative Elevation']) # Filter for desired metric

best_distance = grouped_by_year_and_month['Cummulative Distance'].max()
best_distance_year = grouped_by_year_and_month[grouped_by_year_and_month['Cummulative Distance'] == best_distance]['year'].unique()[0]

best_elevation = grouped_by_year_and_month['Cummulative Elevation'].max()
best_elevation_year = grouped_by_year_and_month[grouped_by_year_and_month['Cummulative Elevation'] == best_elevation]['year'].unique()[0]




grouped_by_year_and_month = grouped_by_year_and_month[grouped_by_year_and_month['type'].isin([activity_type])]
grouped_by_year_and_month = grouped_by_year_and_month[grouped_by_year_and_month['year'].isin(selected_year)]


fig = px.line(grouped_by_year_and_month, x='month', y=selected_metric, color='year')
fig.update_layout(
        xaxis=dict(
            showline=True,
            showgrid=False,
            showticklabels=True,
            linecolor='rgb(204, 204, 204)',
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family='Arial',
                size=12,
                color='rgb(82, 82, 82)',
            ),
        ),
        yaxis=dict(
            # showgrid=True,
            zeroline=False,
            showline=True,
            gridcolor = 'rgb(235, 236, 240)',
            showticklabels=True,
            title='',
            autorange=True
        ),
        autosize=True,
        hovermode="x unified",
        # margin=dict(
        #     autoexpand=True,
        #     l=100,
        #     r=20,
        #     t=110,
        # ),
        showlegend=False,
#         legend=dict(
#         # orientation="h",
#         yanchor="bottom",
#         y=0.9,
#         xanchor="left",
#         x=0.7
# ),
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title='',
        yaxis_title='miles' if selected_metric == 'Cummulative Distance' else 'feet',
        margin=dict(l=0, r=0, t=0, b=0)
    )
fig.for_each_trace(lambda trace: fig.add_annotation(
    x=trace.x[-1], y=trace.y[-1], text='  '+trace.name, 
    font_color=trace.line.color,
    ax=10, ay=10, xanchor="left", showarrow=False))
fig.update_traces(mode="markers+lines", hovertemplate=None)

st.plotly_chart(fig, use_container_width=True)


##################
# Best and Goals #
##################

col1, col2 = st.columns(2)

with col1:
    st.metric(f'Yearly Distance Best {best_distance_year}', "{:,}".format(best_distance) + ' miles')
with col2:
    st.metric(f'Yearly Elevation Best {best_elevation_year}', "{:,}".format(best_elevation) + ' feet')

