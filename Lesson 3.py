import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
# 1 - getting the data
df = sns.load_dataset('titanic')
print(df.head())
print(df.describe())
print(df.info())
# 2 - drop columns we dont need
#df2 = df.drop(['alive'])
df2 = df.copy()
for col in df.select_dtypes(['object']):
    df2[col] = df[col].astype('category')

######### creating Dummy varibles #########

for col in df2.select_dtypes(['category']):
    df2[col] = df2[col].cat.codes


#print(df2.info())

# 3 - Cleaning all the data : drop all nulls or replacing nan values with average that corresponds to the group

mask = df2['age'] >=18
mean_over_18 = df2.loc[mask,'age'].mean()
mean_under_18 = df2.loc[df2['age'] < 18, 'age'].mean()

df2['age'] = df2.apply(lambda row :mean_over_18 if (pd.isna(row['age']) and row['adult_male']) else (mean_under_18 if pd.isna(row['age']) else row['age']), axis= 1 )

print(df2.info())
# 4 - creating X and y:
df2 = df2.drop(columns=['alive','who','embark_town'])
X= df2.drop('survived', axis= 1) # instead of axis= we can do columns= "column_name"
#X=pd.get_dummies(X)
y = df2['survived']

######## 5 - spliting the data

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,train_size=0.7)

# 6- print the rmse accuracy
print('RMSE Baseline accuracy: ', y_train.std())

# 7.1 linearregression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

#model = LinearRegression()
#model = DecisionTreeRegressor(min_samples_leaf=16)# - this model has problem of overfitting
model = RandomForestRegressor(max_features=15,max_leaf_nodes=8)
model = model.fit(X_train,y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

def RMSE(y, y_pred):
    return ((y - y_pred) ** 2).mean() ** 0.5

print(f'Train RMSE: {RMSE(y_train, y_train_pred):.3f}')
print(f'Test RMSE: {RMSE(y_test, y_test_pred):.3f}')

#8 - look at the feature importance
dic = dict(zip(model.feature_names_in_, model.feature_importances_))
fi = pd.Series(model.feature_importances_, index=model.feature_names_in_)
fi = fi.sort_values(ascending=False)
## we can look with parameters that are not important
print(fi)
plt.figure(figsize=(9, 9))
sns.heatmap(df2.corr(), annot=True, fmt = "0.2f")
plt.show()

## Part 2 :
#EDA
#sns.barplot(x='adult_male', y ='survived', data=df2)
#sns.barplot(x='sex', y ='survived', data=df)
sns.lineplot(x='age_decade', y='survived', data=df.assign(age_decade=df.age.round(-1)))

#plt.show()
############################################# shap module #############################################
import shap
# Initialize SHAP explainer
#shap.initjs()
#explainer = shap.Explainer(model)
#explanation = explainer(X_test)  # New style
#shap.summary_plot(explanation, X_test) # gives a summery graph of the data
#shap.plots.waterfall(explanation[0]) # How much does the parameters is important to classify specific prediction
#shap.plots.partial_dependence('age',model.predict , X_test) # by changing the age value lets see how much does the survival rate changes
#shap.plots.partial_dependence('fare', model.predict, X_test, feature_names=X_test.columns)
#shap.plots.bar(explanation) - the imporatnce of the parameters


#######################################################################################################################################



#### Assigment : ####
import requests
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
request = requests.get(url)
request.raise_for_status()
with open('adult.csv', 'w') as f:
    f.write(request.text)



description = """\
age: continuous.
workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
fnlwgt: continuous.
education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
education-num: continuous.
marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
sex: Female, Male.
capital-gain: continuous.
capital-loss: continuous.
hours-per-week: continuous.
native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
"""

# creatinf the columns names:

columns = [line.split(':')[0] for line in description.splitlines()] + ['income']
columns = [column.replace('-', '_') for column in columns]
df = pd.read_csv('adult.csv', names=columns)
print(df.head())
print(df.info())

for col in df.select_dtypes('object'):
    df[col] = df[col].astype('category')

for col in df.select_dtypes('category'):
    df[col] = df[col].cat.codes
df = df.dropna()

x= df.drop('income', axis=1)
Y= df['income']

x_train,x_test, Y_train,Y_test = train_test_split(x,Y, train_size=0.7, random_state=42)
print(df.head())
print(df.info())
print(Y_train.std())
print("the test std is : ",Y_test.std())
model_ = RandomForestRegressor()
model_ = model_.fit(x_train,Y_train)
Y_train_pred = model_.predict(x_train)
Y_test_pred = model_.predict(x_test)

train_error = RMSE(Y_train,Y_train_pred)
test_error = RMSE(Y_test,Y_test_pred)

important = pd.Series(model_.feature_importances_, index=model_.feature_names_in_)
important = important
print(important)
print(f'the trainde RMSE is : {train_error} and the test RMSE is : {test_error} ')
#sns.pairplot(data=df,hue='income') # THIS GRAPH GIVES AS A LOT OF GRAPH FOR ALL 2 VARIBALES LIKE A MATRIX
#plt.show()

############################# Isolation Forest #############################
from sklearn.ensemble import IsolationForest

# Load the diamonds dataset
diamonds = sns.load_dataset("diamonds")
df_ = sns.load_dataset('diamonds')

# Drop the 'price' column for training and handle categorical variables
X = pd.get_dummies(diamonds.drop(columns=['price']), drop_first=True)
y = diamonds['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Isolation Forest model
iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
iso_forest.fit(X_train)

# Predict anomaly scores and labels on a copy of X_test
X_test_copy = X_test.copy()
X_test_copy['anomaly_score'] = iso_forest.decision_function(X_test)  # Quantitative weirdness - NEGATIVE SCORE MEAN THAT THE VALUE IS AN ANNOMALY
X_test_copy['anomaly'] = iso_forest.predict(X_test)  # Binary anomaly label
X_test_copy['anomaly_label'] = X_test_copy['anomaly'].map({1: 'Normal', -1: 'Anomaly'})

###
df_['anomaly_score'] = iso_forest.decision_function(X)  # Quantitative weirdness
df_['anomaly'] = iso_forest.predict(X)  # Binary anomaly label
df_['anomaly_label'] = df_['anomaly'].map({1: 'Normal', -1: 'Anomaly'})

###

# Add price and carat back to the copy for plotting
X_test_copy['price'] = y_test.values
X_test_copy['carat'] = diamonds.loc[X_test.index, 'carat']

# Plot carat vs. price with hue as anomaly score (quantitative weirdness)
plt.figure(figsize=(12, 8))
scatter = plt.scatter(
    X_test_copy['carat'], X_test_copy['price'], c=X_test_copy['anomaly_score'], cmap='coolwarm', edgecolor='k'
)
plt.colorbar(scatter, label='Anomaly Score (Weirdness)')
plt.title('Carat vs. Price with Anomaly Score as Hue')
plt.suptitle('Anomaly Score: lower is more anomalous')
plt.xlabel('Carat')
plt.ylabel('Price')
plt.grid(True)
#plt.show()
# AFTER LOOKING AT THE GRAPH FROM THE ISO FORREST WE CAN CHOOSE WHICH VALUES WE CAN MOVE FROM OUR RANDOM FORREST
#sns.displot(data=X_test_copy, x='anomaly_score', kde=True)
model_dymonds  = RandomForestRegressor()
model_dymonds = model_dymonds.fit(X_train,y_train)

train_pred_reg = model_dymonds.predict(X_train)
test_pred_reg = model_dymonds.predict(X_test)

train_rmse = RMSE(y_train,train_pred_reg)
test_rmse = RMSE(y_test,test_pred_reg)

print(f'the tarin rmse is : {train_rmse} and the test rmse is : {test_rmse}')
#### now lets move the anomalies
df2 = df_[df_.anomaly==1].copy()  # only "good" diamonds
X2 = df2.drop(columns=['price', 'anomaly_label', 'anomaly', 'anomaly_score'])
X2 = pd.get_dummies(X2, drop_first=True)
y2 = df2['price']
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.3, random_state=42)

model2 = RandomForestRegressor()
model2.fit(X2, y2)
y2_test_pred = model2.predict(X2_test)
print("Test RMSE:", RMSE(y2_test, y2_test_pred) )


################# Working with dates #################
import requests
import zipfile

# URL of the ZIP file
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip"

# Download the ZIP file
response = requests.get(url)
with open('bikes.zip', "wb") as f:
    f.write(response.content)

# Extract the contents of the ZIP file
with zipfile.ZipFile('bikes.zip') as zip_ref:
    zip_ref.extractall('bikes')

df = pd.read_csv('bikes/hour.csv', index_col='instant')
df = df.drop(columns=['weekday', 'workingday', 'holiday', 'mnth', 'yr', 'season'])  # to demo using dates

# turn the string to a datetime object
df['dteday'] = pd.to_datetime(df['dteday'])

def train_model(df):
    X = df.drop(columns=['cnt', 'casual', 'registered'])
    y = df['cnt']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)


    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print("Train RMSE:", RMSE(y_train, y_train_pred))
    print("Test RMSE:", RMSE(y_test, y_test_pred))

# train the model without using dates at all
train_model(df.drop(columns=['dteday']))

# train the model using dates as integer (number of seconds since 1970)
train_model(df.assign(dteday=df['dteday'].astype('int64')))

# turn the date into features: year, month, day, day of week, holiday
df2 = df.copy()
df2['year'] = df2['dteday'].dt.year
df2['month'] = df2['dteday'].dt.month
df2['day'] = df2['dteday'].dt.day
df2['day_of_week'] = df2['dteday'].dt.dayofweek

# train the model using dates as integer (number of seconds since 1970)
model = train_model(df2.drop(columns=['dteday']))

df_past = df2[df2.year == 2011]
df_future = df2[df2.year == 2012]

X = df2.drop(columns=['cnt', 'casual', 'registered', 'dteday'])
y = df2['cnt']
X_train, X_test, y_train, y_test = X.loc[df_past.index], X.loc[df_future.index], y.loc[df_past.index], y.loc[df_future.index]

model_dates = RandomForestRegressor()
model_dates.fit(X_train, y_train)

y_train_pred = model_dates.predict(X_train)
y_test_pred = model_dates.predict(X_test)

print("Train RMSE:", RMSE(y_train, y_train_pred))
print("Test RMSE:", RMSE(y_test, y_test_pred))

display(pd.Series(model_dates.feature_importances_, model_dates.feature_names_in_).sort_values(ascending=False))
####### Clasification #######
####### DecisionTreeClassifier #######
from sklearn.tree import DecisionTreeClassifier

df = sns.load_dataset('titanic')
df = df.dropna()


X, y = df.drop(columns=['survived', 'alive']), df['survived']
X = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model_titan = DecisionTreeClassifier()
model_titan.fit(X_train, y_train)
y_test_pred = model_titan.predict(X_test)


#y_test == y_test_pred # True predictions - to find out how good is my predictions
print('Correct predictions:', (y_test == y_test_pred).sum(), f", {(y_test == y_test_pred).mean():.2%}")
print('Incorrect predictions:', (y_test != y_test_pred).sum(), f", {(y_test != y_test_pred).mean():.2%}")

def calc_classification_metrics(y_test, y_test_pred):
    # Calculate TP, TN, FP, FN
    TP = ((y_test == 1) & (y_test_pred == 1)).sum()  # True Positives
    TN = ((y_test == 0) & (y_test_pred == 0)).sum()  # True Negatives
    FP = ((y_test == 0) & (y_test_pred == 1)).sum()  # False Positives
    FN = ((y_test == 1) & (y_test_pred == 0)).sum()  # False Negatives

    # Print the results
    print(f"True Positives (TP): {TP}")
    print(f"True Negatives (TN): {TN}")
    print(f"False Positives (FP): {FP}")
    print(f"False Negatives (FN): {FN}")

    # Additional summary
    accuracy = (TP + TN) / len(y_test)
    print(f"Accuracy: {accuracy:.2%}")

    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    # Generate the confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix Heatmap')
    plt.show()
