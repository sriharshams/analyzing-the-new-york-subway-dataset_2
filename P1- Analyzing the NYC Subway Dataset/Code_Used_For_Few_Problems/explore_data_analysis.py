__author__ = 'sms'
import numpy as np
import pandas
import matplotlib.pyplot as plt

def entries_histogram():
    '''
    Before we perform any analysis, it might be useful to take a
    look at the data we're hoping to analyze. More specifically, let's
    examine the hourly entries in our NYC subway data and determine what
    distribution the data follows. This data is stored in a dataframe
    called turnstile_weather under the ['ENTRIESn_hourly'] column.

    Let's plot two histograms on the same axes to show hourly
    entries when raining vs. when not raining. Here's an example on how
    to plot histograms with pandas and matplotlib:
    turnstile_weather['column_to_graph'].hist()

    Your histograph may look similar to bar graph in the instructor notes below.

    You can read a bit about using matplotlib and pandas to plot histograms here:
    http://pandas.pydata.org/pandas-docs/stable/visualization.html#histograms

    You can see the information contained within the turnstile weather data here:
    https://www.dropbox.com/s/meyki2wl9xfa7yk/turnstile_data_master_with_weather.csv
    '''

    turnstile_weather = pandas.read_csv("C:/learning/Udacity/Intro to Data Scinece/intro_to_ds_programming_files/project_3/plot_histogram/turnstile_data_master_with_weather.csv")

    plt.figure()
    turnstile_weather[turnstile_weather['rain']==0]['ENTRIESn_hourly'].hist(bins=180, label="No rain") # your code here to plot a historgram for hourly entries when it is raining
    turnstile_weather[turnstile_weather['rain']==1]['ENTRIESn_hourly'].hist(bins=180, label="Rain") # your code here to plot a historgram for hourly entries when it is not raining
    plt.suptitle("Histogram of Entries per hour")
    plt.xlabel("Entries per hour")
    plt.xlim(xmax=6000)
    plt.ylim(ymax=45000)
    plt.ylabel("Frequency")
    plt.legend()
    return plt

plt = entries_histogram()
plt.show()
