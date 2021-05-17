import pandas as pd
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import sweetviz as sv
from scipy import stats

import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Get all months data and combine them
data_files = glob.glob("./data/source_data/2018/*.csv")
dfs = list()
for filename in data_files:
    df = pd.read_csv(filename)
    dfs.append(df)
df = pd.concat(dfs, axis=0, ignore_index=True)

# Save the combined data into one csv
df.to_csv("./data/output/2018data.csv")

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

# df = df[['date', 'Start station number', 'count']]
# df.columns = ['date', 'station', 'count']

# Extrat date features
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

# Get stations lat and lng to get tempreture
# Data from https://gbfs.capitalbikeshare.com/gbfs/en/station_information.json
stations_info = pd.read_json('./data/station_information.json')
# Convet to data frame
stations_info = pd.DataFrame(stations_info['data']['stations'])

stations_info['short_name'].unique().astype(int)

stations_info = stations_info[['short_name', 'lat', 'lon', 'region_id', 'capacity']]
stations_info = stations_info.rename(columns={'short_name' : 'station'})
stations_info['station'] = stations_info['station'].astype(int)

stations_info

df = pd.merge(df, stations_info, on="station")

# Replace stations ids with index number
stations_ids = sorted(df['station'].unique())
stations_ids = {k: v for v, k in enumerate(stations_ids)}
stations_ids
df = df.replace({'station': stations_ids})
df.head()

# Visualize data
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

# Working day
df.workingday.value_counts()
sns.catplot(x='workingday',data=df,kind='count',height=5,aspect=1)

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

fig, axes = plt.subplots(5, 2)
fig.set_size_inches(12, 20)
sns.boxplot(data=df,y="count",orient="v",ax=axes[0][0])
sns.boxplot(data=df,y="count",x="season",orient="v",ax=axes[0][1])
sns.boxplot(data=df,y="count",x="holiday",orient="v",ax=axes[1][0])
sns.boxplot(data=df,y="count",x="weekend",orient="v",ax=axes[1][1])
sns.boxplot(data=df,y="count",x="workingday",orient="v",ax=axes[2][0])
sns.boxplot(data=df,y="count",x="day_of_week",orient="v",ax=axes[2][1])
sns.boxplot(data=df,y="count",x="day",orient="v",ax=axes[3][0])
sns.boxplot(data=df,y="count",x="month",orient="v",ax=axes[3][1])
sns.boxplot(data=df,y="count",x="hour",orient="v",ax=axes[4][0])
sns.boxplot(data=df,y="count",x="capacity",orient="v",ax=axes[4][1])

axes[0][0].set(ylabel='Count', title="Box Plot On Count")
axes[0][1].set(ylabel='Count', xlabel='Season', title="Box Plot On Count Across Season")
axes[1][0].set(ylabel='Count', xlabel='Holiday', title="Box Plot On Count Across Holiday")
axes[1][1].set(ylabel='Count', xlabel='Weekend', title="Box Plot On Count Across Weekend")
axes[2][0].set(ylabel='Count', xlabel='Working Day', title="Box Plot On Count Across Working Day")
axes[2][1].set(ylabel='Count', xlabel='Day Of Week', title="Box Plot On Count Across Day Of Week")
axes[3][0].set(ylabel='Count', xlabel='Day Of Month', title="Box Plot On Count Across Day Of Month")
axes[3][1].set(ylabel='Count', xlabel='Month', title="Box Plot On Count Across Month")
axes[4][0].set(ylabel='Count', xlabel='Hour Of The Day', title="Box Plot On Count Across Hour Of The Day")
axes[4][1].set(ylabel='Count', xlabel='Hour Of The Day', title="Box Plot On Count Across Capacity")

# Remove outliers
df.describe()
df = df[np.abs(df['count']-df['count'].mean())<=(3*df['count'].std())]


# df['count'].value_counts()[:25]
# df = df.loc[df['count'] < 10]

# outliers = stats.zscore(df['count']).apply(lambda x: np.abs(x) == 3)
# df_without_outliers = df[~outliers]

# df[np.abs(df['count'].mean())<=(3*df['count'].std())]

# Q1 = df.quantile(0.25)
# Q3 = df.quantile(0.80)
# IQR = Q3 - Q1
# trueList = ~((df[['count']] < (Q1 - 1.5 * IQR)) | (df[['count']] > (Q3 + 1.5 * IQR)))

df.describe()
df.head(500)

#corelation matrix.
cor_mat= df[:].corr()
mask = np.array(cor_mat)
mask[np.tril_indices_from(mask)] = False
fig=plt.gcf()
fig.set_size_inches(30,12)
sns.heatmap(data=cor_mat,mask=mask,square=True,annot=True,cbar=True)

# Visualize data using sweetviz library (output will appear in an html file)
my_report = sv.analyze(df)
my_report.show_html() # Default arguments will generate to "SWEETVIZ_REPORT.html"

# Setup the data for the models
# rescale the features
scaler = MinMaxScaler()

# apply scaler() to all the numeric columns 
categorical_vars = ['station', 'hour', 'day', 'month', 'day_of_week', 'season', 'weekend', 'holiday', 'workingday']
# numeric_vars = ['station', 'hour', 'day', 'month', 'day_of_week', 'season']
df[numeric_vars] = scaler.fit_transform(df[numeric_vars])
df.head()
df.describe()

for v in categorical_vars:
    df[v] = df[v].astype('category')

df.info()

column_set = ColumnTransformer([('encoder', OneHotEncoder(),categorical_vars)], remainder='passthrough') 
   
onehot_data = np.array(column_set.fit_transform(df), dtype = np.str) 

df.pop('lat')
df.pop('lon')
y = df.pop('count')
X = df

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size = 0.3, random_state=40)



numeric_features = ['capacity', 'station']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_features = ['hour', 'day', 'month', 'day_of_week', 'season', 'weekend', 'holiday', 'workingday', 'region_id']
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Append classifier to preprocessing pipeline.
# Now we have a full prediction pipeline.
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestRegressor(n_estimators=100))])

clf.fit(X_train, y_train)
print("model score: %.3f" % clf.score(X_test, y_test))

y_pred = clf.predict(X_test)
r2 = sklearn.metrics.r2_score(y_test, y_pred)
print(r2)


# Creating the models
# Linear regression with 3-fold cross validation
folds = KFold(n_splits = 3, shuffle = True, random_state = 100)

lrModel = LinearRegression()
scores = cross_val_score(lrModel, X_train, y_train, scoring='r2', cv=folds)
scores


clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LinearRegression())])

clf.fit(X_train, y_train)
print("model score: %.3f" % clf.score(X_test, y_test))


lrModel.fit(X_train, y_train)

# predict prices of X_test
y_pred = clf.predict(X_test)
r2 = sklearn.metrics.r2_score(y_test, y_pred)
print(r2)


rfModel = RandomForestRegressor(n_estimators=100)
# yLabelsLog = np.log1p(yLabels)
rfModel.fit(X_train,y_train)
y_pred = rfModel.predict(X_test)
r2 = sklearn.metrics.r2_score(y_test, y_pred)
# print ("RMSLE Value For Random Forest: ",rmsle(np.exp(yLabelsLog),np.exp(preds),False))
print ("RMSLE Value For Random Forest: ", r2)


clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(7,5,2), random_state=1))])

clf = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(7,5,2), random_state=1)
# 3-K Fold cross-validation
scores = cross_val_score(clf, X_train, y_train, cv=3)
clf.fit(X_train, y_train)
clf_y_pred = clf.predict(X= X_test)
r2 = sklearn.metrics.r2_score(y_test, clf_y_pred)
print('With hidden neurons of 5 we get an accuracy of', r2)

errors = abs(y_pred - y_test)
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
