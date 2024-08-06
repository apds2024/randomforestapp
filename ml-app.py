import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# ---------------------------------#
# Page layout
# Page expands to full width
st.set_page_config(page_title='A Random Forest Walk',
                   layout='wide')

# ---------------------------------#
    
# Sidebar - Specify parameter settings
with st.sidebar.header('Set Parameters'):
    split_size = st.sidebar.slider(
        'Data split ratio (% for Training Set)', 10, 90, 80, 5)

with st.sidebar.subheader('1. Learning Parameters'):
    parameter_n_estimators = st.sidebar.slider(
        'Number of estimators (n_estimators)', 0, 1000, 100, 100)
    parameter_max_features = st.sidebar.select_slider(
        'Max features (max_features)', options=['sqrt', 'log2'])
    parameter_min_samples_split = st.sidebar.slider(
        'Minimum number of samples required to split an internal node (min_samples_split)', 1, 10, 2, 1)
    parameter_min_samples_leaf = st.sidebar.slider(
        'Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)

with st.sidebar.subheader('2. General Parameters'):
    parameter_random_state = st.sidebar.slider(
        'Seed number (random_state)', 0, 1000, 42, 1)
    #parameter_criterion = st.sidebar.radio(
        #'Performance measure (criterion)', options=['friedman_mse', 'absolute_error'])
    parameter_bootstrap = st.sidebar.select_slider(
        'Bootstrap samples when building trees (bootstrap)', options=[True, False])
    parameter_oob_score = st.sidebar.select_slider(
        'Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[False, True])
    parameter_n_jobs = st.sidebar.select_slider(
        'Number of jobs to run in parallel (n_jobs)', options=[1, -1])
    
# ---------------------------------#
# Model building


def build_model(df):
    X = df.iloc[:, :-1]  # Using all column except for the last column as X
    Y = df.iloc[:, -1]  # Selecting the last column as Y

    # Identify categorical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    numerical_cols = X.select_dtypes(include=['number']).columns

    # Preprocessing for categorical features
    categorical_preprocessor = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Preprocessing for numerical features
    numerical_preprocessor = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_preprocessor, categorical_cols),
            ('num', numerical_preprocessor, numerical_cols)
        ])

    # Data splitting
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=(100-split_size)/100)
    
    st.markdown('**1.2. Data splits**')
    st.write('Training set')
    st.info(X_train.shape)
    st.write('Test set')
    st.info(X_test.shape)

    st.markdown('**1.3. Variable details**:')
    st.write('X variable')
    st.info(list(X.columns))
    st.write('Y variable')
    st.info(Y.name)

    # Fit the preprocessing pipeline
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Define the model
    # Create the parameter dictionary
    rf_params = {
        'n_estimators': parameter_n_estimators,
        'random_state': parameter_random_state,
        'max_features': parameter_max_features,
        'criterion': 'absolute_error',  # Use 'squared_error' or 'poisson' if needed
        'min_samples_split': parameter_min_samples_split,
        'min_samples_leaf': parameter_min_samples_leaf,
        'bootstrap': parameter_bootstrap,
        'oob_score': parameter_oob_score,
        'n_jobs': parameter_n_jobs
    
    }
    rf = RandomForestRegressor(**rf_params)
    rf.fit(X_train_processed, y_train)

    # Predict on train and test data
    y_pred_train = rf.predict(X_train_processed)
    y_pred_test = rf.predict(X_test_processed)

    st.subheader('2. Model Performance')

    # Evaluate the model
    st.write('**2.1. Train set**')
    st.write('Coefficient of determination ($R^2$):')
    st.info(r2_score(y_train, y_pred_train))

    #st.write('Error (MSE):')
    #st.info(mean_squared_error(y_train, y_pred_train))

    st.write('Error (MAE):')
    st.info(mean_absolute_error(y_train, y_pred_train))

    st.markdown('**2.2. Test set**')
    st.write('Coefficient of determination ($R^2$):')
    st.info(r2_score(y_test, y_pred_test))

    #st.write('Error (MSE):')
    #st.info(mean_squared_error(y_test, y_pred_test))

    st.write('Error (MAE):')
    st.info(mean_absolute_error(y_test, y_pred_test))

    st.subheader('3. Model Parameters')
    st.write(rf.get_params())


# ---------------------------------#
st.write("""
# A Random Forest Walk

This app uses a *RandomForestRegressor()* algorithm to build a regression model.

The RandomForestRegressor is an ensemble learning method that builds multiple decision trees
from random subsets of the training data and averages their predictions to improve accuracy and reduce overfitting.
It uses bootstrapping (sampling with replacement) and random feature selection to create diverse trees. 

This approach results in a more robust and accurate prediction model compared to individual decision trees.

Try adjusting the hyperparameters!

""")

# ---------------------------------#

# Main panel

# Displays the dataset
st.subheader('Dataset')

#if uploaded_file is not None:
   # df = pd.read_csv(uploaded_file)
    #st.markdown('**1.1. Glimpse of dataset**')
    #st.write(df)
    #build_model(df)
#else:
#st.info('Awaiting for CSV file to be uploaded.')
#st.button('Press to use Ames Housing Dataset')

ames = fetch_openml(name="house_prices", as_frame=True)
X = pd.DataFrame(ames.data, columns=ames.feature_names)
Y = pd.Series(ames.target, name='SalePrice')
df = pd.concat([X, Y], axis=1)

st.markdown("""The Ames housing dataset is used as the example.
            Change the model parameters on the left to run different models.""")
st.write(df.head(5))

build_model(df)
