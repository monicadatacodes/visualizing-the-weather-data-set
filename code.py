# --------------
# Import the required Libraries
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import calendar
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# Generate a line chart that visualizes the readings in the months

def line_chart(df,period,col):
    """ A line chart that visualizes the readings in the months
    
    This function accepts the dataframe df ,period(day/month/year) and col(feature), which plots the aggregated value of the feature based on the periods. Ensure the period labels are properly named.
    
    Keyword arguments:
    df - Pandas dataframe which has the data.
    period - Period of time over which you want to aggregate the data
    col - Feature of the dataframe
    
    """
    weat = df.groupby(pd.Grouper(level=period, freq='M'))[col].mean()    
    plt.plot(weat.index.month_name(), weat.values)
    plt.title('Timeline in 2012(in months)')
    # Setting X-axis label
    plt.xlabel('Months') 
    # Setting Y-axis label
    plt.ylabel(col)  
    plt.xticks(rotation=90)   
    plt.show()  
    
    







# Function to perform univariate analysis of categorical columns
def plot_categorical_columns(df):
    """ Univariate analysis of categorical columns
    
    This function accepts the dataframe df which analyzes all the variable in the data and performs the univariate analysis using bar plot.
    
    Keyword arguments:
    df - Pandas dataframe which has the data.
    
    """
    cat_weather_df = df.select_dtypes(include=['object']).copy()
    weather_counts = cat_weather_df['Weather'].value_counts()
    plt.figure(figsize = (16,7))
    sns.set(style="darkgrid")
    sns.barplot(weather_counts.index, weather_counts.values, alpha=0.9)
    plt.xticks(rotation = 90)
    plt.title('Frequency Distribution of Weather')
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel('------Weather------', fontsize=12)
    plt.show()







# Function to plot continous plots
def plot_cont(df,plt_typ):
    """ Univariate analysis of Numerical columns
    
    This function accepts the dataframe df, plt_type(boxplot/distplot) which analyzes all the variable in the data and performs the univariate analysis using boxplot or distplot plot.
    
    Keyword arguments:
    df - Pandas dataframe which has the data.
    plt_type - type of plot through which you want to visualize the data
    
    """
    num_weather_df = df.select_dtypes(include=['number']).columns.tolist().copy()
    
    if plt_typ == 'boxplot' :
        for i in range(0,len(num_weather_df),2):
            if len(num_weather_df) > i+1:
                plt.figure(figsize=(10,4))
                plt.subplot(121)
                sns.boxplot(df[num_weather_df[i]])
                plt.subplot(122)            
                sns.boxplot(df[num_weather_df[i+1]])
                plt.tight_layout()
                plt.show()

        else:
            sns.boxplot(df[num_weather_df[i]])
    
    else :

        for i in range(0,len(num_weather_df),2):
            if len(num_weather_df) > i+1:
                plt.figure(figsize=(10,4))
                plt.subplot(121)
                sns.distplot(df[num_weather_df[i]])
                plt.subplot(122)            
                sns.distplot(df[num_weather_df[i+1]])
                plt.tight_layout()
                plt.show()   
            else:
                sns.distplot(df[num_weather_df[i]])     







# Function to plot grouped values based on the feature
def group_values(df,col1,agg1,col2):
    """ Agrregate values by grouping
    
    This function accepts a dataframe, 2 column(feature) and aggregated function(agg1) which groupby the dataframe based on the column and plots the bar plot.
   
    Keyword arguments:
    df - Pandas dataframe which has the data.
    col1 - Feature of the dataframe on which values will be aggregated.
    agg1 - Dictionary of aggregate functions with feature as the key and func as the value
    col2 - Feature of the dataframe to be plot against grouped data.
    
    Returns:
    grouping - Dataframe with all columns on which it is grouped on.
    """
    aggregate = {'mean':np.mean,'max':np.max,'min':np.min}
    grouping = df.groupby(col1).agg(aggregate[agg1])
    plt.figure(figsize=(10,4))
    plt.ylabel(col2)
    grouping[col2].plot(kind="bar")
    plt.show()





# Read the Data and pass the parameter as parse_dates=True, index_col='Date/Time'
weather_df = pd.read_csv(path, parse_dates= True, index_col = 'Date/Time') 




# Lets try to generate a line chart that visualizes the temperature readings in the months.
# Call the function line_chart() with the appropriate parameters.
line_chart(weather_df,'Date/Time','Temp (C)') 


# Now let's perform the univariate analysis of categorical features.
# Call the "function plot_categorical_columns()" with appropriate parameters.
plot_categorical_columns(weather_df)


# Let's plot the Univariate analysis of Numerical columns.
# Call the function "plot_cont()" with the appropriate parameters to plot distplot
plot_cont(weather_df,'boxplot')


# Call the function "plot_cont()" with the appropriate parameters to plot boxplot
plot_cont(weather_df,'distplot')

# Groupby the data by Weather and plot the graph of the mean visibility during different weathers. Call the function group_values to plot the graph.
# Feel free to try on diffrent features and aggregated functions like max, min.
group_values(weather_df,'Weather',"mean",'Visibility (km)')



