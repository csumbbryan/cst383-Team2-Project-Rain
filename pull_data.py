# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 20:24:48 2025

@author: zanol
"""

import datetime
import time
import os
import math
import pandas as pd
import numpy as np
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeRegressor
import graphviz

# key = "bdce6144fd068f935b7a4d78c610dc2a"
# lat = '39.936369'
# lon = '-122.780911'
# start = time.mktime(datetime.datetime(2014, 1, 1, 0, 0, 0).timetuple())

# end = time.mktime(datetime.datetime(2024, 12, 31, 12, 59, 0).timetuple())

# url = f"https://history.openweathermap.org/data/2.5/history/city?lat={lat}&lon={lon}&type=hour&start={start}&end={end}&appid={key}"
# print(url)

# url2 = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&units=imperial&appid={key}"
# print(url2)

# response = requests.get(url2)
# print(response.json())

def rmse(predicted, actual):
    return np.sqrt(((predicted - actual)**2).mean())

##READ IN WEATHER STATION DATA FROM ALL CSVs
folder_path = r'C:\Users\bryan.zanoli\OneDrive\Documents\School\CSUMB\CST383-30\project\station_data'

df_weatherdata = pd.DataFrame()

for filename in os.listdir(folder_path):
    if os.path.isfile(os.path.join(folder_path, filename)):
        df_station = pd.read_csv(folder_path + "\\" + filename, nrows = 1, header = 0)
        df_weather = pd.read_csv(folder_path + "\\" + filename, skiprows = 3)
        #print('Station: ',df_station.head(3))
        #print('Weather: ',df_weather.head(3))
        df_weather['key'] = 1
        df_station['key'] = 1
        df_new = pd.merge(df_station, df_weather, on='key').drop('key', axis = 1)
        #print('New: ',df_new.head(3))
        df_weatherdata = pd.concat([df_new, df_weatherdata], axis = 0)

df_weatherdata['year'] = pd.to_datetime(df_weatherdata['time']).dt.year

print('Weather Data Read In: ',df_weatherdata.head(10))
df_weatherdata.info()

##CREATE coordinates DataFrame from df_weatherdata
df_location = df_weatherdata[['latitude', 'longitude']].drop_duplicates()
print('Location Data Read In: \n',df_location)

##READ IN TEST DATA
folder_path = r'C:\Users\bryan.zanoli\OneDrive\Documents\School\CSUMB\CST383-30\project\test_data_with_prec'

df_testdata = pd.DataFrame()

for filename in os.listdir(folder_path):
    if os.path.isfile(os.path.join(folder_path, filename)):
        df_station = pd.read_csv(folder_path + "\\" + filename, nrows = 1, header = 0)
        df_weather = pd.read_csv(folder_path + "\\" + filename, skiprows = 3)
        #print('Station: ',df_station.head(3))
        #print('Weather: ',df_weather.head(3))
        df_weather['key'] = 1
        df_station['key'] = 1
        df_new = pd.merge(df_station, df_weather, on='key').drop('key', axis = 1)
        #print('New: ',df_new.head(3))
        df_testdata = pd.concat([df_new, df_testdata], axis = 0)

df_testdata['year'] = pd.to_datetime(df_testdata['time']).dt.year
df_testdata['location'] = df_testdata['latitude'].astype('str')+' '+df_testdata['longitude'].astype('str')

print('Test Data Read In: ',df_testdata.head(10))
df_testdata.info()

df_testdata['temp_max_mean'] = df_testdata.groupby(['location', 'year'])['temperature_2m_max (°F)'].transform(np.mean)
df_testdata['temp_min_mean'] = df_testdata.groupby(['location', 'year'])['temperature_2m_min (°F)'].transform(np.mean)
df_testdata['temp_min_ann'] = df_testdata.groupby(['location', 'year'])['temperature_2m_min (°F)'].transform(np.min)
df_testdata['temp_max_ann'] = df_testdata.groupby(['location', 'year'])['temperature_2m_max (°F)'].transform(np.max)
df_testdata['temp_mean_ann'] = df_testdata.groupby(['location', 'year'])['temperature_2m_mean (°F)'].transform(np.mean)
df_testdata['temp_mean_max'] = df_testdata.groupby(['location', 'year'])['temperature_2m_mean (°F)'].transform(np.max)
df_testdata['temp_mean_min'] = df_testdata.groupby(['location', 'year'])['temperature_2m_mean (°F)'].transform(np.min)
df_testdata['wind_speed_max_min'] = df_testdata.groupby(['location', 'year'])['wind_speed_10m_max (mp/h)'].transform(np.min)
df_testdata['wind_speed_max_max'] = df_testdata.groupby(['location', 'year'])['wind_speed_10m_max (mp/h)'].transform(np.max)
df_testdata['wind_speed_max_mean'] = df_testdata.groupby(['location', 'year'])['wind_speed_10m_max (mp/h)'].transform(np.mean)
df_testdata['wind_gusts_max_min'] = df_testdata.groupby(['location', 'year'])['wind_gusts_10m_max (mp/h)'].transform(np.min)
df_testdata['wind_gusts_max_max'] = df_testdata.groupby(['location', 'year'])['wind_gusts_10m_max (mp/h)'].transform(np.max)
df_testdata['wind_gusts_max_mean'] = df_testdata.groupby(['location', 'year'])['wind_gusts_10m_max (mp/h)'].transform(np.mean)
df_testdata['precipitation_hours_sum'] = df_testdata.groupby(['location', 'year'])['precipitation_hours (h)'].transform(np.sum)
df_testdata['precipitation_hours_mean'] = df_testdata.groupby(['location', 'year'])['precipitation_hours (h)'].transform(np.mean)
df_testdata['evapotranspiration_min'] = df_testdata.groupby(['location', 'year'])['et0_fao_evapotranspiration (inch)'].transform(np.min)
df_testdata['evapotranspiration_max'] = df_testdata.groupby(['location', 'year'])['et0_fao_evapotranspiration (inch)'].transform(np.max)
df_testdata['evapotranspiration_mean'] = df_testdata.groupby(['location', 'year'])['et0_fao_evapotranspiration (inch)'].transform(np.mean)
df_testdata['precipitation_sum'] = df_testdata.groupby(['location', 'year'])['precipitation_sum (inch)'].transform(np.sum)

df_testdata.drop(['time', 'temperature_2m_max (°F)', 'temperature_2m_min (°F)', 'temperature_2m_mean (°F)',
                          'precipitation_hours (h)', 'wind_speed_10m_max (mp/h)', 'wind_gusts_10m_max (mp/h)',
                          'et0_fao_evapotranspiration (inch)', 'rain_sum (inch)', 'precipitation_sum (inch)'], 
            axis = 1, inplace = True)

print(df_testdata['location'][0:8000:500])

df_testdata = df_testdata.drop_duplicates(subset = ['location', 'year'])
df_testdata = df_testdata.reset_index(drop=True)

df_testdata.info()

print(df_testdata)

##CREATE coordinates DataFrame from df_weatherdata
df_location = df_weatherdata[['latitude', 'longitude']].drop_duplicates()
print('Location Data Read In: \n',df_location)


##READ IN RAIN STATION DATA
df_rain = pd.read_csv(r"C:\Users\bryan.zanoli\OneDrive\Documents\School\CSUMB\CST383-30\project\lwu-precip-data-to-2023_basic_flatfile_withcoordinates.csv", index_col=None)
df_rain.columns = df_rain.columns.str.strip()
df_rain.info()

df_rain = df_rain.dropna(subset = ['TotalPrecipitation_inches', 'WaterYear'])
nonfloat = np.where(df_rain['TotalPrecipitation_inches'] == '.')[0]
df_rain.drop(df_rain.index[nonfloat], inplace=True)
df_rain.drop(['BeginGageServiceDate', 'EndGageServiceDate', 'Notes_FlaggedResults'], axis = 1, inplace = True)

df_rain['TotalPrecipitation_inches'] = df_rain['TotalPrecipitation_inches'].astype('float64')

df_rain.info()


##Find nearest WEATHER STATION to RAIN STATION
for rain in df_rain.index :
    min_dist = float(1e7)
    selected_long = 0
    selected_lat = 0
    for location in df_location.values:
        # print('Location: ', location)
        # print('X Coord: ', df_rain.loc[rain, ['x_coord']])
        # print('Operation: ', location[1] - df_rain.loc[rain, ['x_coord']])
        distance = (location[1] - df_rain.loc[rain, ['x_coord']].values)**2 + (location[0] - df_rain.loc[rain, ['y_coord']].values)**2
        if(distance < min_dist):
            min_dist = distance
            selected_long = location[1]
            selected_lat = location[0]
        #print(distances)
    df_rain.loc[rain, 'min_distance'] = min_dist
    df_rain.loc[rain, 'longitude'] = selected_long
    df_rain.loc[rain, 'latitude'] = selected_lat

print(df_rain[['StationName', 'x_coord', 'y_coord', 'longitude', 'latitude']])
print(df_weatherdata[['longitude', 'latitude']])

#Merge WEATHER and RAIN STATION data
df_all = pd.merge(df_rain, df_weatherdata, left_on = ['latitude', 'longitude', 'WaterYear'], right_on=['latitude', 'longitude', 'year'], how = 'inner')
print(df_all[['StationName', 'time', 'WaterYear', 'wind_speed_10m_max (mp/h)']])

#df_all.to_csv(r"C:\Users\zanol\OneDrive\Documents\School\CSUMB\CST383-30\project\output.csv")

##Create new annual aggs based on monthly variables
df_all['temp_max_mean'] = df_all.groupby(['StationName', 'year'])['temperature_2m_max (°F)'].transform(np.mean)
df_all['temp_min_mean'] = df_all.groupby(['StationName', 'year'])['temperature_2m_min (°F)'].transform(np.mean)
df_all['temp_min_ann'] = df_all.groupby(['StationName', 'year'])['temperature_2m_min (°F)'].transform(np.min)
df_all['temp_max_ann'] = df_all.groupby(['StationName', 'year'])['temperature_2m_max (°F)'].transform(np.max)
df_all['temp_mean_ann'] = df_all.groupby(['StationName', 'year'])['temperature_2m_mean (°F)'].transform(np.mean)
df_all['temp_mean_max'] = df_all.groupby(['StationName', 'year'])['temperature_2m_mean (°F)'].transform(np.max)
df_all['temp_mean_min'] = df_all.groupby(['StationName', 'year'])['temperature_2m_mean (°F)'].transform(np.min)
df_all['wind_speed_max_min'] = df_all.groupby(['StationName', 'year'])['wind_speed_10m_max (mp/h)'].transform(np.min)
df_all['wind_speed_max_max'] = df_all.groupby(['StationName', 'year'])['wind_speed_10m_max (mp/h)'].transform(np.max)
df_all['wind_speed_max_mean'] = df_all.groupby(['StationName', 'year'])['wind_speed_10m_max (mp/h)'].transform(np.mean)
df_all['wind_gusts_max_min'] = df_all.groupby(['StationName', 'year'])['wind_gusts_10m_max (mp/h)'].transform(np.min)
df_all['wind_gusts_max_max'] = df_all.groupby(['StationName', 'year'])['wind_gusts_10m_max (mp/h)'].transform(np.max)
df_all['wind_gusts_max_mean'] = df_all.groupby(['StationName', 'year'])['wind_gusts_10m_max (mp/h)'].transform(np.mean)
df_all['precipitation_hours_sum'] = df_all.groupby(['StationName', 'year'])['precipitation_hours (h)'].transform(np.sum)
df_all['precipitation_hours_mean'] = df_all.groupby(['StationName', 'year'])['precipitation_hours (h)'].transform(np.mean)
df_all['evapotranspiration_min'] = df_all.groupby(['StationName', 'year'])['et0_fao_evapotranspiration (inch)'].transform(np.min)
df_all['evapotranspiration_max'] = df_all.groupby(['StationName', 'year'])['et0_fao_evapotranspiration (inch)'].transform(np.max)
df_all['evapotranspiration_mean'] = df_all.groupby(['StationName', 'year'])['et0_fao_evapotranspiration (inch)'].transform(np.mean)


##Drop all monthly only variables as granularity isn't required
df_all_new = df_all.drop(['time', 'temperature_2m_max (°F)', 'temperature_2m_min (°F)', 'temperature_2m_mean (°F)',
                          'precipitation_hours (h)', 'wind_speed_10m_max (mp/h)', 'wind_gusts_10m_max (mp/h)',
                          'et0_fao_evapotranspiration (inch)'], axis = 1)

##Output to CSV (Optional)
#df_all_new.head().to_csv(r"C:\Users\zanol\OneDrive\Documents\School\CSUMB\CST383-30\project\output2.csv")

##Drop duplicate rows as elimination of monthly variables should allow for large reduction
df_all_new = df_all_new.drop_duplicates(subset = ['StationName', 'year'])
df_all_new = df_all_new.reset_index(drop=True)

##Verify new dataframe
df_all_new.info()

#df_all_new.to_csv(r"C:\Users\zanol\OneDrive\Documents\School\CSUMB\CST383-30\project\output_curated.csv")

##
##BRYAN SANDBOX
##
# predictors = ['temp_max_mean', 'temp_min_mean', 'temp_min_ann', 'temp_max_ann', 'temp_mean_ann', 'temp_mean_max',
#               'temp_mean_min', 'wind_speed_max_min', 'wind_speed_max_max', 'wind_speed_max_mean', 'wind_gusts_max_min',
#               'wind_gusts_max_max', 'wind_gusts_max_mean', 'evapotranspiration_min', 'evapotranspiration_max',
#               'evapotranspiration_mean', 'elevation']
predictors = ['elevation', 'latitude', 'longitude']
target = 'TotalPrecipitation_inches'

predictors_max_r2 =['elevation',
                    'StationName_Blacks Mountain',
                    'StationName_Boulder Creek',
                    'StationName_Butte Lake',
                    'StationName_Camel Peak',
                    'StationName_Champs Flat',
                    'StationName_Clarks Peak',
                    'StationName_Clover Valley',
                    'StationName_Crowder Flat',
                    'StationName_Dewitt Peak',
                    'StationName_Dodge Reservoir',
                    'StationName_Gazelle Mountain',
                    'StationName_Granite Springs',
                    'StationName_Hogsback Road',
                    'StationName_Lassen Creek',
                    'StationName_Lights Creek',
                    'StationName_Little Last Chance',
                    'StationName_Long Bell Station',
                    'StationName_McCarthy Point',
                    'StationName_Medicine Lake',
                    'StationName_Mount Hough',
                    'StationName_Mount Shasta',
                    'StationName_Mumbo Basin',
                    'StationName_Onion Valley',
                    'StationName_Patterson Meadow',
                    'StationName_Pepperdine Camp',
                    'StationName_Saddle Camp',
                    'StationName_Shaffer Mountain',
                    'StationName_Stouts Meadow',
                    'StationName_Swain Mountain',
                    'StationName_Sweagert Flat']

X_train = df_all_new[~(df_all_new['year'] == 2023)][predictors]
y_train = df_all_new[~(df_all_new['year'] == 2023)][target]
X_test = df_all_new[df_all_new['year'] == 2023][predictors]
y_test = df_all_new[df_all_new['year'] == 2023][target]
#X = df_all_new[predictors]
#y = df_all_new[target]
X = X_train
y = y_train

print(X.shape)
print(y.shape)

fig, axes = plt.subplots(nrows = math.ceil(np.sqrt(X.shape[1])), ncols = math.ceil(X.shape[1] / math.ceil(np.sqrt(X.shape[1]))),
                         gridspec_kw={'wspace': .4, 'hspace': .9})
plt.rcParams['axes.titlesize'] = 6
plt.figure(figsize= (20, 15))
plt.tight_layout()

i = 0
cols = math.ceil(X.shape[1] / math.ceil(np.sqrt(X.shape[1])))
for predictor in predictors:
    k = i % cols
    j = math.floor (i / cols)
    X_ = X[[predictor]]
    X_train_ = X_train[[predictor]]
    X_test_ = X_test[[predictor]]
    # X_train, X_test, y_train, y_test = train_test_split(X_, y, test_size=0.25, random_state=42)
    reg = LinearRegression()
    reg.fit(X_train_, y_train)
    predicted = reg.predict(X_test_)
    axes[j][k].scatter(X_, y, s = 2)
    axes[j][k].scatter(X_test_, predicted, s = 2)
    axes[j][k].set_title(predictor)
    axes[j][k].tick_params(axis='both', labelsize=6)
    i += 1
    print('R2 Score of {}: {:.3f}'.format(predictor, r2_score(y_test, predicted)))



#plt.subplots_adjust(wspace=2, hspace=2, left=1, right=1.1, bottom=1, top=1.1)
plt.show()

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
reg.fit(X_train, y_train)
predicted = reg.predict(X_test)
print('R2 Score of All: {:.3f}'.format(r2_score(y_test, predicted)))

sns.scatterplot(x = predicted, y = y_test, s = 10, label = 'Test')
sns.scatterplot(x = reg.predict(X_train), y = y_train, s = 10, label = 'Train')
plt.legend()
plt.show()

##FEATURE ANALYSIS GET DUMMIES ON STATION NAME
df_all_dum = pd.get_dummies(df_all_new, columns = ['StationName'], dtype = 'int')
df_all_dum.info()

X_train = df_all_dum[~(df_all_dum['year'] == 2023)][predictors_max_r2]
y_train = df_all_dum[~(df_all_dum['year'] == 2023)][target]
X_test = df_all_dum[df_all_dum['year'] == 2023][predictors_max_r2]
y_test = df_all_dum[df_all_dum['year'] == 2023][target]

regdum = LinearRegression()
regdum.fit(X_train, y_train)
predict = regdum.predict(X_test)
print(predict)
print(y_test)
print(y_test.index)
print(r2_score(y_test, predict))
for i in range(len(y_test)):
    print('Station Location: ', df_all_dum.loc[y_test.index[i]][['latitude', 'longitude']].values)
    print('  Predicted: {:.2f} ||  Actual: {:.2f}'.format(predicted[i], y_test.iloc[i]))
print(regdum.coef_)

##Polynomial Features Analysis
X = df_all_new[df_all_new['year'] == 2023][predictors]
y = df_all_new[df_all_new['year'] == 2023][target]
pf = PolynomialFeatures(degree = 3)
pf.fit(X)
X_poly = pf.transform(X)
print(X_poly.shape)
print(X_poly)

rmse_min = float(1e7)
rmse_i_min = 0
r2_max = float(0)
r2_i_max = 0
rmse_selected = [[1e7 for _ in range(2)] for _ in range(10)]
r2_selected = [[0 for _ in range(2)] for _ in range(10)]

for i in range(X_poly.shape[1]):
    X_ = X_poly[:,[i]]
    X_train, X_test, y_train, y_test = train_test_split(X_, y, test_size=0.25, random_state=42)
    
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    predicted = reg.predict(X_test)
    rmse_train = rmse(reg.predict(X_train), y_train)
    rmse_test = rmse(predicted, y_test)
    r2 = r2_score(y_test, predicted)
    if rmse_test < rmse_min:
        rmse_min = rmse_test
        rmse_i_min = i
        print('New RMSE Test Min: {:.3f}'.format(rmse_min))
        for j in range(10):
            if rmse_min < rmse_selected[j][0]:
                rmse_selected[j][0] = rmse_min
                rmse_selected[j][1] = rmse_i_min
                break
    if r2 > r2_max:
        r2_max = r2
        r2_i_max = i
        print('New R2 Max: {:.3f}'.format(r2_max))
        for j in range(10):
            if r2_max > r2_selected[j][0]:
                r2_selected[j][0] = r2_max
                r2_selected[j][1] = r2_i_max
                break
print(rmse_selected)
print(r2_selected) 

print(pf.get_feature_names_out()[18]) 

X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.25, random_state=42)

remaining = list(range(X_train.shape[1]))
i_min = 0
selected = []
n = 10

while len(selected) < n: 
    rmse_min = 1e7
    for i in remaining: 
        current = selected.copy()
        current.append(i)
        X_ = X_train[:, current]
        scores = cross_val_score(LinearRegression(), X_, y_train, scoring = 'neg_mean_squared_error', cv = 5)
        rmse_value = np.sqrt(-scores.mean())
        if (rmse_value < rmse_min):
            rmse_min = rmse_value
            i_min = i
    remaining.remove(i_min)
    selected.append(i_min)
    print('num features: {}; rmse: {:.2f}'.format(len(selected), rmse_min))

for feat_num in selected:
    print('Feature Names: ', pf.get_feature_names_out()[feat_num])  
    
X_10 = X_poly[:, [18]]
X_train, X_test, y_train, y_test = train_test_split(X_10, y, test_size=0.25, random_state=42)
reg2 = LinearRegression()
reg2.fit(X_train, y_train)
print(r2_score(y_test, reg2.predict(X_test)))

##TESTING THE TRUE TEST SET USING ABOVE
X_t = df_testdata[df_testdata['year'] == 2023][predictors]
y_t = df_testdata[df_testdata['year'] == 2023]['precipitation_sum']
#pf.fit(X_t)
X_t = pf.transform(X_t)
X_t = X_t[:, [18]]
(print(X_t.shape))
predicted = reg2.predict(X_t)
print(predicted)
print(y_t)
print(reg2.coef_)


#DECISION TREE REGRESSOR - JUST FOR FUN

X_dt = df_all_new[predictors]
y_dt = df_all_new[target]


tree_reg = DecisionTreeRegressor(max_depth=3)
tree_reg.fit(X_dt, y_dt)

dot_data = export_graphviz(
        tree_reg, precision = 2,
        feature_names=predictors,
        rounded=True,
        filled=True
    )
graph = graphviz.Source(dot_data)
graph.view()

print(X_t)

print(tree_reg.predict(X_t))
print(y_t)


