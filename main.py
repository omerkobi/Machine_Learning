# ML class :
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
df =sns.load_dataset('iris')
#print(sns.pairplot(data=df, hue='species'))
#plt.show()
# lets "shuffle" the penguins so that we mix them properly
df = df.sample(frac=1)
print(len(df))
train_n = int(len(df) * 0.7)
df_train = df.iloc[:train_n]
df_test = df.iloc[train_n:]

sns.displot(data=df_train, x='petal_length', hue='species', kind='kde')
#plt.show()
#Based on the displot kde graph we filter each species for prediction
def predict_iris_species_petal_length(row):
    if row.petal_length<2.5:
        return 'setosa'
    elif row.petal_length < 4.8:
        return 'versicolor'
    else:
        return 'virginica'

train_pred = df_train.apply(predict_iris_species_petal_length, axis=1)
print(train_pred)
test_pred =  (train_pred == df_train.species).mean()# Accuracy 97%
print(test_pred) # 95% accuracy

###############
def predict_iris_species_petal_width(row):
    if row.petal_width<0.75:
        return 'setosa'
    elif row.petal_width < 1.6:
        return 'versicolor'
    else:
        return 'virginica'

train_pred = df_train.apply(predict_iris_species_petal_width, axis=1)
test_width=(train_pred == df_train.species).mean() # Accuracy 96%
print(test_width)

test_pred = df_test.apply(predict_iris_species_petal_width, axis=1)
test_test = (test_pred == df_test.species).mean()

print("for test:",test_test ) # Accuracy 91%

# for sepal lengh
sns.displot(data=df_train, x='sepal_length', hue='species', kind='kde')
def predict_iris_species_sepal_length(row):
    if row.sepal_length<5.5:
        return 'setosa'
    elif row.sepal_length < 6.5:
        return 'versicolor'
    else:
        return 'virginica'

train_pred = df_train.apply(predict_iris_species_sepal_length, axis=1)
print((train_pred == df_train.species).mean())  # Accuracy 75%

print((test_pred == df_test.species).mean())

# Now we can cross all of the function to finds multiple True in order to filter correctly for species

def predict_iris_species(row):
    predictors = [predict_iris_species_sepal_length, predict_iris_species_petal_width, predict_iris_species_petal_length]
    results = pd.Series([predictor(row) for predictor in predictors]).value_counts()
    if max(results) == 1:  # they disagree, so take the one most accuracte
        return predict_iris_species_petal_length(row)
    else:
        return results.index[0] # the index of the species that got the most vote - beacuse value_counts is sorted from biggest to smallest

test_pred = df_test.apply(predict_iris_species, axis=1)
print((test_pred == df_test.species).mean())  # Accuracy

########## How to automate this to be machine learning:

###################### THE ZERO MODEL ######################
'''first Lets switch to an easier problem: predicting the weights of penguins

this is easier problem because we can average the predictions of "experts"'''

df = sns.load_dataset('penguins')
prediction = df.body_mass_g.mean()
'''the "zero" model (the simplest model that has some value) is the model that predicts the average

this is actual the stddev standard deviation

we will use this as the baseline - more complicated models have to be (significantly) bette'''

ste =(((df.body_mass_g - prediction)**2).mean())**0.5 # the standart deveation
print(ste)

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
#from scikit-learn.tree import
from sklearn.tree import export_graphviz
import graphviz

def visualize_first_levels(tree, max_depth=2):
    # Export the tree to dot format
    dot_data = export_graphviz(
        tree,
        out_file=None,
        feature_names=model.feature_names_in_,
        filled=True,
        rounded=True,
        special_characters=True,
        max_depth=max_depth  # Limit the depth
    )

    # Use graphviz to visualize the tree
    graph = graphviz.Source(dot_data)
    return graph



df = sns.load_dataset('penguins')
df = df.dropna()                 # drop null values
df = df.drop(columns=['species', 'island', 'sex'])  # remove categorical variables we dont (yet!) know how to handle
#print(df.body_mass_g.head(20))
df = df.sample(frac=1) # df becames random sample


train_n = int(len(df) * 0.7)
df_train = df.iloc[:train_n]
df_test = df.iloc[train_n:]

model = DecisionTreeRegressor()
X = df.drop(columns='body_mass_g')  # the data without the answer
y = df.body_mass_g
X_train = X.iloc[:train_n]
y_train = y.iloc[:train_n]

model.fit(X_train, y_train)  # create a tree from the
#visualize_first_levels(model, )


# Call the visualization function
#graph = visualize_first_levels(model, )
#graph.render("decision_tree", format="png", cleanup=True)  # Save to a PNG file
#graph.view()  # Open the rendered graph
#plt.show()

y_train_pred = model.predict(X_train)

def root_mean_squared_error(y_true, y_pred):
    return ((y_true - y_pred)**2).mean() ** 0.5

print("hello")
print(root_mean_squared_error(y_train, y_train_pred))
X_test = X.iloc[train_n:]
y_test = y.iloc[train_n:]

model = DecisionTreeRegressor(min_samples_leaf=4) #The minimum number of samples required to be at a leaf node
model.fit(X_train, y_train)  # create/learn a tree from the data
visualize_first_levels(model, )

y_train_pred = model.predict(X_train) # the prediction for the training data
y_test_pred = model.predict(X_test) # the prediction for the texting data
#print(X_test.iloc[0])
print(y_test_pred)
print(root_mean_squared_error(y_train, y_train_pred), root_mean_squared_error(y_test, y_test_pred))

'''its a function from data to a number/category
the "machine learning" part is trying to FIND this function, from the almost limiless number of functions that exist, such that it has a low error
we fit/train/grow it on "training" data where is sees the input and the answer
we test it on test/validation data where we give it the input without the output,
 and then we need to calculate the error with some metric (for regression, most often using RMSE Root Mean Squared Error)'''
##################### summary #####################
'''take a dataset
clean it by removing or changing elements that machine learning models dont understand - Nulls and strings
choose a machine learning model (in this module we choose RandomForest) and its hyperparameters
split the data df into X and y
split X,y into train and `test
fit the model using X_train and y_train
predict both X_train and X_test and measure the'''

'''parameters VS hyperparameters:

in machine learning:

the parameters are the "decisions" in the model (i.e. the decision tree itself, or if doing linear algebra the values in the matrix)
the hyperparameters are the parameters to the LEARNING function that FINDS the model.
'''

############ functions for cleaning ##################
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def extreme_data_cleaning(df):
    """
    Drops all rows with null values and all columns with non-numeric values.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    # Step 1: Drop all rows with null values.
    # WARNING: This removes any row with at least one null value.
    # Alternative: Use df.dropna(subset=[columns], how='all') to drop selectively.
    df = df.dropna()

    # Step 2: Select only numeric columns.
    # WARNING: This will discard all non-numeric columns, even if they are important.
    # Alternative: Use feature engineering to encode categorical variables instead.
    df = df.select_dtypes(include=['number'])

    return df

from dataclasses import dataclass
#### know lets do tree reggression :
@dataclass
class ModelResults:
    model: object
    X: pd.DataFrame
    y: pd.Series
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series

def train_and_evaluate(df, target_column, model_type="tree", model_kwargs=None):
    """
    Cleans the data, splits it into training and test sets, trains a model, and evaluates it.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_column (str): The name of the target column.
        model_type (str): The type of model to use ("tree" or "random_forest").
        model_kwargs (dict): Additional hyperparameters for the model.

    Returns:
        ModelResults: A dataclass containing the model, features, and training/testing datasets.
    """
    if model_kwargs is None:
        model_kwargs = {}

    # Step 1: Clean the data using extreme_data_cleaning
    cleaned_df = extreme_data_cleaning(df)

    # Step 2: Split into features (X) and target (y)
    X = cleaned_df.drop(columns=[target_column])
    y = cleaned_df[target_column]

    # Step 3: Perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 4: Initialize the model
    if model_type == "tree":
        model = DecisionTreeRegressor(**model_kwargs)
    elif model_type == "random_forest":
        model = RandomForestRegressor(**model_kwargs)
    else:
        raise ValueError("Invalid model_type. Choose 'tree' or 'random_forest'.")

    # Step 5: Fit the model
    model.fit(X_train, y_train)

    # Step 6: Make predictions and calculate RMSE
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    print(f"Train RMSE: {train_rmse}")
    print(f"Test RMSE: {test_rmse}")

    # Recommendations for improvement:
    # 1. Instead of extreme cleaning, handle missing values with imputation (e.g., mean, median, or mode).
    # 2. Encode categorical variables using one-hot encoding or label encoding.
    # 3. OPTIONAL Use feature scaling (e.g., StandardScaler or MinMaxScaler) to standardize data for certain models.
    # 4. Perform hyperparameter tuning using GridSearchCV or RandomizedSearchCV for better model optimization.
    # 5. Explore feature selection techniques (e.g., correlation matrix or feature importance) to reduce dimensionality and improve performance.

    # Step 7: Return results - saving them inside a class

    return ModelResults(
        model=model,
        X=X,
        y=y,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test
    )

##########################################
def plot_regression_results(results: ModelResults):
    """
    Generates useful plots for regression analysis using the ModelResults.

    Args:
        results (ModelResults): The ModelResults object containing the model and datasets.
    """
    # Residual plot
    y_train_pred = results.model.predict(results.X_train)
    y_test_pred = results.model.predict(results.X_test)

    residuals_train = results.y_train - y_train_pred
    residuals_test = results.y_test - y_test_pred

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_train_pred, y=residuals_train, label="Train", alpha=0.7)
    sns.scatterplot(x=y_test_pred, y=residuals_test, label="Test", alpha=0.7)
    plt.axhline(0, color='red', linestyle='--')
    plt.title("Residuals vs. Predictions")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.legend()
    plt.show()

    # Feature importance plot (only for tree-based models)
    if hasattr(results.model, "feature_importances_"):
        feature_importances = pd.DataFrame({
            'Feature': results.X.columns,
            'Importance': results.model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importances)
        plt.title("Feature Importances")
        plt.show()

    # Recommendations for improvement:
    # 1. Use cross-validation instead of a single train-test split for more robust evaluation.
    # 2. Analyze additional plots, such as learning curves or prediction error distribution.
    # 3. Consider using SHAP or LIME for detailed explainability of feature contributions.

    # Actual vs Predicted plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=results.y_test, y=y_test_pred, alpha=0.7)
    plt.plot([results.y_test.min(), results.y_test.max()], [results.y_test.min(), results.y_test.max()], color='red', linestyle='--')
    plt.title("Actual vs. Predicted Values")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.show()


df = sns.load_dataset('diamonds')
#results = train_and_evaluate(df, 'price', model_kwargs={'min_samples_leaf': 16})
#print(plot_regression_results(results))

print(sns.get_dataset_names())

df2 = sns.load_dataset('mpg')
#lets do acceleration as y varible
pd.set_option('display.max_columns', None)
print(len(df2))
print(df2.tail())

result = train_and_evaluate(df2,'acceleration',model_kwargs={'min_samples_leaf': 16})
# display graphs for the regression
#print(plot_regression_results(result))

#print(result.model.predict(result.X_test))
#print(result.X_test.head(20))
#print('hello')
#print(df2.iloc[248])
#df3 = df2.drop(columns=['acceleration'])
#print(result.model.predict(df3))

###################################### RANDOM FOREST ######################################
from sklearn.metrics import mean_squared_error

def load_and_prepare_data(test_size=0.2, random_state=42):
    """
    Loads the seaborn diamonds dataset, does basic cleaning/encoding,
    and performs a train/test split.

    Returns:
        X_train, X_test, y_train, y_test
    """
    # Load dataset
    df = sns.load_dataset('diamonds')

    # We will predict 'price' using other features
    X = df.drop('price', axis=1)
    y = df['price']

    # Convert categorical columns to dummy variables
    X = pd.get_dummies(X, drop_first=True)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = load_and_prepare_data() #getting the data

def train_decision_tree_default(X_train, y_train):
    """
    Trains a DecisionTreeRegressor with default hyperparameters.

    Returns:
        model: trained DecisionTreeRegressor
    """
    model = DecisionTreeRegressor()  # all default
    model.fit(X_train, y_train)
    return model

def train_decision_tree_improved(X_train, y_train, random_state=42):
    """
    Trains a DecisionTreeRegressor with some "improved" hyperparameters
    (this is just an example; you can tweak these).

    Returns:
        model: trained DecisionTreeRegressor
    """
    model = DecisionTreeRegressor(
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model

def train_decision_tree_random_splitter(X_train, y_train, random_state=42): # split the tree by random conditions
    """
    Trains a DecisionTreeRegressor with splitter='random'.

    Returns:
        model: trained DecisionTreeRegressor
    """
    model = DecisionTreeRegressor(splitter='random', random_state=random_state)
    model.fit(X_train, y_train)
    return model


def train_and_ensemble_random_trees(
    X_train, y_train, X_test, y_test,
    max_trees=20,
    random_state=42
):
    """
    Trains multiple DecisionTreeRegressor models with splitter='random',
    each with a different random_state to get randomness in splits,
    then averages their predictions and computes the test RMSE.

    Returns:
        tree_counts: list of number of trees (1..max_trees)
        rmses: list of RMSEs corresponding to ensemble sizes
    """
    # Keep track of predictions from each tree so we can average later
    all_test_predictions = []
    tree_counts = []
    rmses = []

    np.random.seed(random_state)  # for reproducibility of random seeds
    for i in range(1, max_trees + 1):
        # train a new random-split tree with a random random_state
        rs = np.random.randint(0, 10_000)  # random seed each time
        model = DecisionTreeRegressor(splitter='random', random_state=rs)
        model.fit(X_train, y_train)

        # Predict on the test set
        test_pred = model.predict(X_test)
        all_test_predictions.append(test_pred)

        # Average predictions from all models so far
        ensemble_pred = np.mean(all_test_predictions, axis=0) # axis = 0 - mean that the mean should be along the rows - mean of all the fisr , second values in the lists

        # Calculate RMSE
        rmse = mean_squared_error(y_test, ensemble_pred) ** 0.5

        tree_counts.append(i)
        rmses.append(rmse)

    return tree_counts, rmses

def plot_ensemble_rmse(tree_counts, rmses):
    """
    Plots the RMSE vs number of trees in the ensemble.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(tree_counts, rmses, marker='o')
    plt.title("Test RMSE vs Number of Random-Split Trees in the Ensemble")
    plt.xlabel("Number of Trees in Ensemble")
    plt.ylabel("Test RMSE")
    plt.grid(True)
    plt.show()

#tree_counts, rmses = train_and_ensemble_random_trees(X_train, y_train, X_test, y_test, max_trees=20)
#plot_ensemble_rmse(tree_counts, rmses)

from sklearn.ensemble import RandomForestRegressor

''' how can we improve the random forrest model? 
we wont split the trees randomly - we will give each model diffrent data sets to train appon
(diffrent precentages of the data)'''
import time
start_time = time.time() # the start time for the execution
# try diffrent values for max_features
test_random = RandomForestRegressor(n_estimators=100,max_features= 10,min_samples_leaf=2, min_samples_split=2)
test_random.fit(X_train,y_train)

test_rand_pred = test_random.predict(X_train)
test_rand_test = test_random.predict(X_test)

train_rmse = mean_squared_error(y_train,test_rand_pred) ** 0.5
test_rmse = mean_squared_error(y_test,test_rand_test) ** 0.5

end_time = time.time() # the end time for the execution

#print(f'the train rmse is : {train_rmse} ,  the test rmse is : {test_rmse}Execution time: {end_time - start_time:.4f} secoonds ' )

##### Dealing with catagorical data  : ##########
df_ =sns.load_dataset('titanic')
X = df_.drop(columns='fare')
y = df_['fare']
pd.get_dummies(X[['class']])  # dummy columns, one-hot encoding
############
pd.get_dummies(X)  # for each string column, create column for each value
############
# dir(X['class'].cat)
#['class'].cat.categories
X['class_cat'] = X['class'].cat.codes
print(X.head())
# converting catagorical string into a category
def encode_all_categories(df):
    df = df.copy()
    for col in df.select_dtypes(include='category').columns:
        df[col] = df[col].cat.codes

    return df

# using my knowledge of this particular dataset, I encode string columns
# that
for col in ['class', 'sex', 'embarked', 'embark_town', 'alive', 'who']:
    X[col] = X[col].astype('category')
encode_all_categories(X)







