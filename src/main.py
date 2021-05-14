import pandas as pd
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import sweetviz as sv

import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# Get all 2019 months data
data_files = glob.glob("./data/2018/*.csv")
dfs = list()
for filename in data_files:
    df = pd.read_csv(filename)
    dfs.append(df)
df = pd.concat(dfs, axis=0, ignore_index=True)

# Save the combined data into one csv
df.to_csv("./data/2019data.csv")

df.info()
df.head()

# Check if there is null values
df.isnull().sum()

# Get holidays in US
min_date = df.min()['Start date'].split()[0]
max_date = df.max()['Start date'].split()[0]
holidays = USFederalHolidayCalendar().holidays(start=min_date, end=max_date)

holidays

# Convert dates from str to datetime
df['date']= pd.to_datetime(df['Start date'])
# Set all dates minutes and seconds to 0
df['date'] = df['date'].dt.floor('H')
# Initial count
df['count'] = 1

df = df.groupby(['date','Start station number'])[['count']].sum().reset_index()
df.columns = ['date', 'station', 'count']

# Extrat new features
df['hour'] = df['date'].dt.hour
df['day'] = df['date'].dt.day
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek + 1
df['weekend'] = (df['day_of_week'] >= 5).astype(int)
df['season'] = df['date'].dt.month%12 // 3 + 1
df['holiday'] = df['date'].dt.date.astype('datetime64').isin(holidays).astype(int)
df['workingday'] = (~(df['weekend'].astype(bool) | df['holiday'].astype(bool))).astype(int)

df.head(100)

# Remove date column
df.pop('date')

df.describe()

df['count'].value_counts()[:20]
df = df.loc[(df['count'] < 15)]

# From https://gbfs.capitalbikeshare.com/gbfs/en/station_information.json
stations_info = pd.read_json('./data/station_information.json')
# Convet to data frame
stations_info = pd.DataFrame(stations_info['data']['stations'])

stations_info['short_name'].unique().astype(int)

stations_info = stations_info[['short_name', 'lat', 'lon']]
stations_info = stations_info.rename(columns={'short_name' : 'station'})
stations_info['station'] = stations_info['station'].astype(int)

df = pd.merge(df, stations_info, on="station")

# Seasons
df.season.value_counts()
sns.catplot(x='season',data=df,kind='count',height=5,aspect=1)

# Holidays
df.holiday.value_counts()
sns.catplot(x='holiday',data=df,kind='count',height=5,aspect=1)
# Discard (Biasing)

# Weekend
df.weekend.value_counts()
sns.catplot(x='weekend',data=df,kind='count',height=5,aspect=1)

# Day of week
df.day_of_week.value_counts()
sns.catplot(x='day_of_week',data=df,kind='count',height=5,aspect=1)

# Day of month
df.day.value_counts()
sns.catplot(x='day',data=df,kind='count',height=5,aspect=1)

# Month
df.month.value_counts()
sns.catplot(x='month',data=df,kind='count',height=5,aspect=1)

# Hour
df.hour.value_counts()
sns.catplot(x='hour',data=df,kind='count',height=5,aspect=1)

fig, axes = plt.subplots(nrows=2,ncols=2)
fig.set_size_inches(12, 10)
sns.boxplot(data=df,y="count",orient="v",ax=axes[0][0])
sns.boxplot(data=df,y="count",x="season",orient="v",ax=axes[0][1])
sns.boxplot(data=df,y="count",x="hour",orient="v",ax=axes[1][0])
sns.boxplot(data=df,y="count",x="workingday",orient="v",ax=axes[1][1])

axes[0][0].set(ylabel='Count',title="Box Plot On Count")
axes[0][1].set(xlabel='Season', ylabel='Count',title="Box Plot On Count Across Season")
axes[1][0].set(xlabel='Hour Of The Day', ylabel='Count',title="Box Plot On Count Across Hour Of The Day")
axes[1][1].set(xlabel='Working Day', ylabel='Count',title="Box Plot On Count Across Working Day")

df = df[np.abs(df['count']-df['count'].mean())<=(3*df['count'].std())] 
print ("Shape Of The Before Ouliers: ",df.shape)
print ("Shape Of The After Ouliers: ",dfWithoutOutliers.shape)


# rescale the features
scaler = MinMaxScaler()

# apply scaler() to all the numeric columns 
# numeric_vars = ['count', 'hour', 'day', 'day_of_week', 'season']
numeric_vars = ['station']
df[numeric_vars] = scaler.fit_transform(df[numeric_vars])
df.head()

my_report = sv.analyze(df)
my_report.show_html() # Default arguments will generate to "SWEETVIZ_REPORT.html"

#corelation matrix.
cor_mat= df[:].corr()
mask = np.array(cor_mat)
mask[np.tril_indices_from(mask)] = False
fig=plt.gcf()
fig.set_size_inches(30,12)
sns.heatmap(data=cor_mat,mask=mask,square=True,annot=True,cbar=True)

df.pop('station')
y = df.pop('count')
X = df

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size = 0.3, random_state=40)

# create a KFold object with 3 splits 
folds = KFold(n_splits = 3, shuffle = True, random_state = 100)

lrModel = LinearRegression()
scores = cross_val_score(lrModel, X_train, y_train, scoring='r2', cv=folds)
scores

lrModel.fit(X_train, y_train)

# predict prices of X_test
y_pred = lrModel.predict(X_test)
r2 = sklearn.metrics.r2_score(y_test, y_pred)
print(r2)


rfModel = RandomForestRegressor(n_estimators=100)
# yLabelsLog = np.log1p(yLabels)
rfModel.fit(X_train,y_train)
y_pred = rfModel.predict(X_test)
r2 = sklearn.metrics.r2_score(y_test, y_pred)
# print ("RMSLE Value For Random Forest: ",rmsle(np.exp(yLabelsLog),np.exp(preds),False))
print ("RMSLE Value For Random Forest: ", r2)


clf = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(7,5,2), random_state=1)
# 3-K Fold cross-validation
scores = cross_val_score(clf, X_train, y_train, cv=3)
clf.fit(X_train, y_train)
clf_y_pred = clf.predict(X= X_test)
r2 = sklearn.metrics.r2_score(y_test, clf_y_pred)
print('With hidden neurons of 5 we get an accuracy of', r2)

errors = abs(clf_y_pred - y_test)
# Calculate mean absolute percentage error (MAPE)
mape = np.mean(100 * (errors / y_test))
# Calculate and display accuracy
accuracy = 100 - mape    
#print result
print('The best model achieves on the test set an accuracy of', round(accuracy, 2),'%')

time = '2020-01-17 10:35:20'
test = rfModel.predict([[31918, 12, 9, 6, 1, 4, 1]])
print(test)
df


sns.catplot(x="day_of_week",y='count',kind='bar',data=df,height=5,aspect=1)

df['start_station'] = df['start_station'].str.replace('[^A-Za-z\s0-9]+', '')
df.head(10)

#To get all unique stations
stationlist = list(df['Start_station'].unique())
len(stationlist)
df['start_station'].value_counts

df.head(100)
df['address_1'] = (df['start_station'] + ', District of Columbia')
value_counts = df['address_1'].value_counts()
df_val_counts = pd.DataFrame(value_counts)
df = df_val_counts.reset_index()
df.columns = ['address_1', 'counts'] 
stationlist = list(df['address_1'].unique())

df