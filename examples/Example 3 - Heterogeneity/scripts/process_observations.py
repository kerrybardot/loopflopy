import pandas as pd
from shapely.geometry import LineString,Point,Polygon,MultiPolygon,shape
import matplotlib.pyplot as plt


# Now we have git extra bore details from WIR, we can groundlevel and well screens to our dataframe
def make_df():
    # Add GL to main df
    df = pd.read_excel('../data/bore_data.xlsx', sheet_name='obs_bores')
    df = df.drop(columns=['Depth From/To (mbGL)'])
    df = df.rename(columns={'Site Short Name': 'ID'})
    df_boredetails = df
    return df_boredetails 

# Now we have bore details, we can add Water Level observations to our dataframe
def add_WL_obs(df_boredetails):
    # Import water level data from WIR
    WL = pd.read_excel('../data/waterlevel_data.xlsx')

    # Filter based on date and variable name
    WL = WL[WL['Collect Date'] > '2005-01-01']
    start_date = '2005-01-01'
    df_filtered = WL[
                    (WL['Collect Date'] >= start_date) & 
                    (WL['Variable Name'] == 'Water level (AHD) (m)')
                    ]

    # Add ID, screened aquifer to main df
    df_obs = pd.merge(df_boredetails, df_filtered, on='Site Ref', how='left')
    df_obs = df_obs[['ID', 'Site Ref', 'Collect Date', 'Aquifer Name', 'Reading Value',]]

    df_boredetails['min_WL'] = None
    df_boredetails['max_WL'] = None
    df_boredetails['mean_WL'] = None

    bores = df_obs['Site Ref'].unique()
    for bore in bores:
        df = df_obs[df_obs['Site Ref'] == bore]
        df_boredetails.loc[df_boredetails['Site Ref'] == bore, 'min_WL'] = df['Reading Value'].min()
        df_boredetails.loc[df_boredetails['Site Ref'] == bore, 'max_WL'] = df['Reading Value'].max()
        df_boredetails.loc[df_boredetails['Site Ref'] == bore, 'mean_WL'] = df['Reading Value'].mean()

    return (df_boredetails, df_obs)

def plot_hydrographs(df_obs):
    # Plot water levels - Yarragadee Aquifer
    yarr_df = df_obs[df_obs['Aquifer Name'] == 'Perth-Yarragadee North']
    yarr_bores = yarr_df['Site Ref'].unique()

    for bore in yarr_bores:
        df = yarr_df[yarr_df['Site Ref'] == bore]
        plt.plot(df['Collect Date'], df['Reading Value'], label = df['ID'].iloc[0])
    plt.legend(loc = 'upper left',fontsize = 'small', markerscale=0.5)
