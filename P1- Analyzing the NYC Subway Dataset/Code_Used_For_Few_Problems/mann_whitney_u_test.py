__author__ = 'sms'
import numpy as np
import scipy
import scipy.stats
import pandas

def mann_whitney_plus_means():
    '''
    This function will consume the turnstile_weather dataframe containing
    our final turnstile weather data.

    You will want to take the means and run the Mann Whitney U-test on the
    ENTRIESn_hourly column in the turnstile_weather dataframe.

    This function should return:
        1) the mean of entries with rain
        2) the mean of entries without rain
        3) the Mann-Whitney U-statistic and p-value comparing the number of entries
           with rain and the number of entries without rain

    You should feel free to use scipy's Mann-Whitney implementation, and you
    might also find it useful to use numpy's mean function.

    Here are the functions' documentation:
    http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html
    http://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html

    You can look at the final turnstile weather data at the link below:
    https://www.dropbox.com/s/meyki2wl9xfa7yk/turnstile_data_master_with_weather.csv
    '''
    #turnstile_weather = pandas.read_csv("C:/learning/Udacity/Intro to Data Scinece/improved-dataset/improved-dataset/turnstile_weather_v2.csv")
    turnstile_weather = pandas.read_csv("C:/learning/Udacity/Intro to Data Scinece/intro_to_ds_programming_files/project_3/mann_whitney_u_test/turnstile_data_master_with_weather.csv")
    ### YOUR CODE HERE ###
    turnstile_weather_no_rain = turnstile_weather[turnstile_weather['rain']==0]
    turnstile_weather_rain = turnstile_weather[turnstile_weather['rain']==1]

    print len(turnstile_weather_rain['ENTRIESn_hourly'])
    print len(turnstile_weather_no_rain['ENTRIESn_hourly'])

    print np.median(turnstile_weather_rain['ENTRIESn_hourly'])
    print np.median(turnstile_weather_no_rain['ENTRIESn_hourly'])

    with_rain_mean = np.mean(turnstile_weather_rain['ENTRIESn_hourly'])
    without_rain_mean = np.mean(turnstile_weather_no_rain['ENTRIESn_hourly'])

    (u, pvalue) = scipy.stats.mannwhitneyu(turnstile_weather_no_rain['ENTRIESn_hourly'], turnstile_weather_rain['ENTRIESn_hourly'] )

    return with_rain_mean, without_rain_mean, u, pvalue*2 # leave this line for the grader

print mann_whitney_plus_means()