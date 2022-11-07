# Imports

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

import streamlit as st

st.set_page_config(
    page_title="Get Around Analysis",
    page_icon="🚗",
    layout="wide"
)


st.title("Get Around Analysis ")
st.markdown("""
Analysis visualizations
""")




### Functions ###
@st.cache(allow_output_mutation=True)
def load_data():
    df = pd.read_excel("get_around_delay_analysis.xlsx")
    return df

def feature_engineering(df):
    # Create a new column without negatives and none values for the delay
    df['delay'] = df['delay_at_checkout_in_minutes'].apply(lambda x : x if x > 0 else 0)

    # Create a new column with the log of the delay
    df['log_delay'] = df['delay'].apply(lambda x : np.log(x))

    # Create a new column to check if the rental is delayed or not
    df['is_delayed'] = df['delay'].apply(lambda x : True if x > 0 else False)

    # List of id of delayed rentals
    delayed_list = df[df['is_delayed'] == True]['rental_id'].tolist()

    # Create a new column to check if previous rental is delayed or not
    df['is_previous_delayed'] = df['previous_ended_rental_id'].apply(lambda x : True if x in delayed_list else False)

    # Create a column that check if there is a next rental
    df['rented_after'] = df['rental_id'].apply(lambda x : True if x in df['previous_ended_rental_id'].to_list() else False)

    return df

def only_ended_rentals(df):
    # Create dataframe for ended rentals
    df_ended = df[df['state'] == 'ended']
    df_ended = df_ended.reset_index(drop=True)
    return df_ended

def removing_delay_outliers(df):
    # removing the delay outliers
    delay_mean = df['delay'].mean()
    delay_std = df['delay'].std()
    df = df[(df['delay'] > delay_mean - 2 * delay_std) & (df['delay'] < delay_mean + 2 * delay_std)]
    df = df.reset_index(drop=True)
    return df

def two_scopes_analysis(df):
    # Create dataframe for connect and mobile
    df_connect = df[df['checkin_type'] == 'connect']
    df_mobile = df[df['checkin_type'] == 'mobile']
    return df_connect, df_mobile



### Sidebar ###


st.sidebar.markdown("# GetAround Analysis Visualizations")


### Visualizations ###

# Create dataframes
ga_df = load_data()
ga_df = feature_engineering(ga_df)
ga_connect_df , ga_mobile_df = two_scopes_analysis(ga_df)

# Display the dataframe
st.header("Dataframe")
with st.expander("See the dataframe"):
    st.write(ga_df)

st.markdown("""
    ***
""")

# Put scopes and delay vizualizations in  2 columns
st.header("Delay Visualizations")
delay_col1, delay_col2 = st.columns(2)

with delay_col1 :
    # Compare the 2 scopes
    st.subheader("Compare the 2 scopes")
    fig_scopes = px.bar(ga_df, x="checkin_type",title="Checkin type")
    fig_scopes_delay = px.histogram(ga_df, x='checkin_type',y='delay')

    st.plotly_chart(fig_scopes)
    # Plotting delay at checkout for connected and mobile in a bar chart
    st.plotly_chart(fig_scopes_delay)

    st.markdown("""
    * More rentals mobile scope than in connect scope.
    * The delay higher in the mobile scope than in the connect scope.
    
    """)


with delay_col2 :
    # Visualization for the delay
    st.subheader("Delay at checkout")
    fig_delay = px.histogram(ga_df, x='delay', title="Delay at checkout")
    st.plotly_chart(fig_delay)

    # Visualization for the log delay
    fig_log_delay = px.histogram(ga_df[ga_df['log_delay'] >= 0], x='log_delay', title="Log delay at checkout")
    st.plotly_chart(fig_log_delay)
    st.markdown("""
    - distribution of delay counts is more visible with the log of delay
    - The most of the delays occures between 19 min and 131 min. Average length of 51 min""")


# Visualization for delayed rentals

delayed_col1, delayed_col2 = st.columns(2)

with delay_col1 :
    # Visualization for the delayed rentals for the connect scope
    st.subheader("Connect scope : delayed rentals")
    fig_proportion_delayed_connect = px.pie(
        ga_connect_df['is_delayed'].value_counts(),
        values='is_delayed',
        names=ga_connect_df['is_delayed'].unique()
    )
    st.plotly_chart(fig_proportion_delayed_connect)


with delay_col2 :
    # Visualization for the delayed rentals for the mobile scope
    st.subheader("Mobile scope : Delayed rentals")
    fig_proportion_delayed_mobile = px.pie(
        ga_mobile_df['is_delayed'].value_counts(),
        values='is_delayed',
        names=ga_mobile_df['is_delayed'].unique()
    )
    st.plotly_chart(fig_proportion_delayed_mobile)

st.markdown("""
    - Mobile rentale : half delayed
    - Connected rentals : 2/3 delayed
    - Connected : more likely to be late at checkout
    """)



# Visualization when previous rental

# Visualization after delayed rental

# Number of rentals rented after a delayed rental
st.subheader("Rentals after a delayed rental")
fig_rented_after = px.pie(
    ga_df['rented_after'].value_counts(),
    values='rented_after',
    names=ga_df['rented_after'].unique(),
    title="Rentals after a delayed rental"
)
st.plotly_chart(fig_rented_after,use_container_width=True)

# Number of rentals rented after a delayed rental for each scope
rented_after_col1, rented_after_col2 = st.columns(2)

with rented_after_col1 :
    # Rentals after a delayed rental for the connect scope
    st.subheader("Rentals after a delayed rental for the connect scope")
    fig_rented_after_connect = px.pie(
        ga_connect_df['rented_after'].value_counts(),
        values='rented_after',
        names=ga_connect_df['rented_after'].unique(),
        title="Rentals after a delayed rental for the connect scope"
    )
    st.plotly_chart(fig_rented_after_connect)

with rented_after_col2 :
    # Rentals after a delayed rental for the mobile scope
    st.subheader("Rentals after a delayed rental for the mobile scope")
    fig_rented_after_mobile = px.pie(
        ga_mobile_df['rented_after'].value_counts(),
        values='rented_after',
        names=ga_mobile_df['rented_after'].unique(),
        title="Rentals after a delayed rental for the mobile scope"
    )
    st.plotly_chart(fig_rented_after_mobile)
st.markdown("""
A large proportion of rentals are not rented after a delayed rental.
""")

# Rentals rented after a rental not delayed
st.subheader("State of rentals that were rented after a rental not delayed")
fig_rented_after_not_delayed = px.pie(
    ga_df[ga_df['is_previous_delayed'] == False]['state'].value_counts(),
    values='state',
    names=ga_df['state'].unique()
)
st.plotly_chart(fig_rented_after_not_delayed,use_container_width=True)
st.markdown("""
More than 80% of previously rented cars are canceled. So delay at checkout may not be the main criteria for cancelation
""")

### Only ended dataframe ###

# Ended rentals

st.header("Only ended dataframe")
ga_df = ga_df[ga_df['state'] == 'ended']
ga_connect_df = ga_connect_df[ga_connect_df['state'] == 'ended']
ga_mobile_df = ga_mobile_df[ga_mobile_df['state'] == 'ended']

# Removing the delay outliers
ga_df = removing_delay_outliers(ga_df)
ga_connect_df = removing_delay_outliers(ga_connect_df)
ga_mobile_df = removing_delay_outliers(ga_mobile_df)

# Display the dataframe
with st.expander("Display data") :
    st.write(ga_df)

st.markdown("""
***
""")

### Scope ###

scope = st.sidebar.selectbox("Chose a scope", ["Connect", "Mobile"])
value = 0
df = pd.DataFrame()

if scope == "Connect":
    # threshold for the delay at the third quartile (for rentals that were ended for connect)
    value = int(ga_connect_df['delay'].quantile(0.75))
    df = ga_connect_df
elif scope == "Mobile":
    # Threshold for the delay at the third quartile (for rentals that were ended for mobile)
    value = int(ga_mobile_df['delay'].quantile(0.75))
    df = ga_mobile_df

### Threshold ###

threshlod = st.sidebar.slider("Threshold", min_value=0, max_value=100, value=value)

### Price per day ###

# Price per day to average by default
price_per_day = st.sidebar.slider("Price per day", min_value=10, max_value=422, value=121)


### Rentals affected by the threshold ###

# Number of rentals ended with a delay above the threshold
st.header("Rentals affected by the threshold")
st.info(
    f'The number of rentals that were ended with a delay above the threshold is {df[df["delay"] > threshlod]["delay"].count()}'
)

st.markdown("""
***
""")


### Revenue affected by the threshold ###

st.header("Share of the revenue that is affected by the threshold")
price_per_minute = price_per_day / (60 * 24)
st.info(
    f'The share of the revenue that is affected by the threshold is {round(price_per_minute * threshlod,2)}€'
)