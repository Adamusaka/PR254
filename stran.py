import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

st.set_page_config(page_title="Napoved cen nepremičnin (Kaggle)", layout="wide")
st.title("Napoved cen nepremičnin – Kaggle Housing Prices")


@st.cache(allow_output_mutation=True)
def load_and_clean_data():

    train_df = pd.read_csv("./Podatki/train.csv")
    test_df  = pd.read_csv("./Podatki/test.csv")
    
    outliers = [598, 955, 935, 1299, 250, 314, 336, 707, 379, 1183, 692, 186, 
                441, 524, 739, 636, 1062, 1191, 496, 198, 1338]
    train_df = train_df[~train_df["Id"].isin(outliers)].reset_index(drop=True)
    
    objectAtrib = ['FireplaceQu', 'MasVnrType', 'Fence', 'Alley', 
                   'GarageCond', 'GarageFinish', 'GarageQual', 
                   'BsmtExposure', 'BsmtQual', 'BsmtCond']
    intAtrib    = ['MasVnrArea', 'LotFrontage']
    unfAtrib    = ['BsmtFinType1', 'BsmtFinType2']

    for col in objectAtrib:
        train_df[col].fillna('No', inplace=True)
        test_df[col].fillna('No', inplace=True)

    for col in intAtrib:
        train_df[col].fillna(0, inplace=True)
        test_df[col].fillna(0, inplace=True)

    for col in unfAtrib:
        train_df[col].fillna('Unf', inplace=True)
        test_df[col].fillna('Unf', inplace=True)

    train_df['Electrical'].fillna('SBrkr', inplace=True)
    test_df['Electrical'].fillna('SBrkr', inplace=True)

    drop_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence',
                 'GarageYrBlt', 'GarageCond', 'BsmtFinType2']
    train_df.drop(columns=drop_cols, inplace=True, errors='ignore')
    test_df.drop(columns=drop_cols, inplace=True, errors='ignore')

    train_df['houseAge']       = train_df['YrSold'] - train_df['YearBuilt']
    test_df['houseAge']        = test_df['YrSold']  - test_df['YearBuilt']
    train_df['houseRemodelAge']= train_df['YrSold'] - train_df['YearRemodAdd']
    test_df['houseRemodelAge'] = test_df['YrSold']  - test_df['YearRemodAdd']
    train_df['totalSF']        = (train_df['1stFlrSF'] + train_df['2ndFlrSF'] +
                                  train_df['BsmtFinSF1'] + train_df['BsmtFinSF2'])
    test_df['totalSF']         = (test_df['1stFlrSF'] + test_df['2ndFlrSF'] +
                                  test_df['BsmtFinSF1'] + test_df['BsmtFinSF2'])
    train_df['totalArea']      = train_df['GrLivArea'] + train_df['TotalBsmtSF']
    test_df['totalArea']       = test_df['GrLivArea']  + test_df['TotalBsmtSF']
    train_df['totalBaths']     = (train_df['BsmtFullBath'] + train_df['FullBath'] +
                                  0.5 * (train_df['BsmtHalfBath'] + train_df['HalfBath']))
    test_df['totalBaths']      = (test_df['BsmtFullBath'] + test_df['FullBath'] +
                                  0.5 * (test_df['BsmtHalfBath'] + test_df['HalfBath']))
    train_df['totalPorchSF']   = (train_df['OpenPorchSF'] + train_df['3SsnPorch'] +
                                  train_df['EnclosedPorch'] + train_df['ScreenPorch'] +
                                  train_df['WoodDeckSF'])
    test_df['totalPorchSF']    = (test_df['OpenPorchSF'] + test_df['3SsnPorch'] +
                                  test_df['EnclosedPorch'] + test_df['ScreenPorch'] +
                                  test_df['WoodDeckSF'])

    drop_after_eng = [
        'Id', 'YrSold', 'YearBuilt', 'YearRemodAdd',
        '1stFlrSF', '2ndFlrSF', 'BsmtFinSF1', 'BsmtFinSF2',
        'GrLivArea', 'TotalBsmtSF', 'BsmtFullBath', 'FullBath',
        'BsmtHalfBath', 'HalfBath', 'OpenPorchSF', '3SsnPorch',
        'EnclosedPorch', 'ScreenPorch', 'WoodDeckSF', 'GarageArea'
    ]
    train_df.drop(columns=drop_after_eng, inplace=True, errors='ignore')
    test_df.drop(columns=drop_after_eng, inplace=True, errors='ignore')

    train_df['SalePrice'] = np.log1p(train_df['SalePrice'])

    return train_df, test_df

train_df, test_df = load_and_clean_data()

@st.cache(allow_output_mutation=True)
def build_preprocessing_pipeline(df: pd.DataFrame):

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.drop("SalePrice")

    ode_cols = [
        'LotShape', 'LandContour', 'Utilities', 'LandSlope',  'BsmtQual',
        'BsmtFinType1', 'CentralAir', 'Functional', 'FireplaceQu',
        'GarageFinish', 'GarageQual', 'PavedDrive', 'ExterCond',
        'KitchenQual', 'BsmtExposure', 'HeatingQC','ExterQual', 'BsmtCond'
    ]

    ohe_cols = [
        'Street', 'LotConfig', 'Neighborhood', 'Condition1', 'Condition2',
        'BldgType', 'HouseStyle', 'RoofStyle', 'Exterior1st', 'Exterior2nd',
        'MasVnrType', 'Foundation', 'Electrical', 'SaleType', 'MSZoning',
        'SaleCondition', 'Heating', 'GarageType', 'RoofMatl'
    ]

    num_pipeline = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="mean")),
        ("scale",  StandardScaler())
    ])

    ode_pipeline = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])

    ohe_pipeline = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", num_pipeline, num_cols),
        ("ode", ode_pipeline, ode_cols),
        ("ohe", ohe_pipeline, ohe_cols),
    ], remainder="passthrough", n_jobs=-1)

    return preprocessor

preprocessor = build_preprocessing_pipeline(train_df)

@st.cache(allow_output_mutation=True)
def train_ensemble_model(df: pd.DataFrame, _preprocessor):
    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]

    X_preproc = preprocessor.fit_transform(X)
    X_train, X_val, y_train, y_val = train_test_split(
        X_preproc, y, test_size=0.2, random_state=25
    )

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_val)
    rmse_lr = np.sqrt(mean_squared_error(y_val, y_pred_lr))

    rfr = RandomForestRegressor(random_state=13)
    param_grid_rfr = {
        "max_depth": [5, 10],
        "n_estimators": [100, 250],
        "min_samples_split": [3, 5]
    }
    rfr_cv = GridSearchCV(rfr, param_grid_rfr, cv=3,
                          scoring="neg_mean_squared_error", n_jobs=-1)
    rfr_cv.fit(X_train, y_train)
    best_rfr = rfr_cv.best_estimator_
    y_pred_rfr = best_rfr.predict(X_val)
    rmse_rfr = np.sqrt(mean_squared_error(y_val, y_pred_rfr))

    xgb = XGBRegressor(random_state=13, eval_metric="rmse", use_label_encoder=False)
    param_grid_xgb = {
        "learning_rate": [0.05, 0.1],
        "n_estimators": [300],
        "max_depth": [3],
        "min_child_weight": [1, 2],
        "gamma": [0, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }
    xgb_cv = GridSearchCV(xgb, param_grid_xgb, cv=2,
                          scoring="neg_mean_squared_error", n_jobs=-1)
    xgb_cv.fit(X_train, y_train)
    best_xgb = xgb_cv.best_estimator_
    y_pred_xgb = best_xgb.predict(X_val)
    rmse_xgb = np.sqrt(mean_squared_error(y_val, y_pred_xgb))

    cat = CatBoostRegressor(loss_function="RMSE", verbose=False, random_state=13)
    param_grid_cat = {
        "iterations": [100, 300],
        "depth": [4, 6],
        "learning_rate": [0.01, 0.05]
    }
    cat_cv = GridSearchCV(cat, param_grid_cat, cv=2,
                          scoring="neg_mean_squared_error", n_jobs=-1)
    cat_cv.fit(X_train, y_train)
    best_cat = cat_cv.best_estimator_
    y_pred_cat = best_cat.predict(X_val)
    rmse_cat = np.sqrt(mean_squared_error(y_val, y_pred_cat))

    vr = VotingRegressor(
        estimators=[
            ("rfr", best_rfr),
            ("xgb", best_xgb),
            ("cat", best_cat)
        ],
        weights=[2, 3, 1]
    )
    vr.fit(X_train, y_train)
    y_pred_vr = vr.predict(X_val)
    rmse_vr = np.sqrt(mean_squared_error(y_val, y_pred_vr))

    estimators_stack = [
        ("rfr", best_rfr),
        ("xgb", best_xgb),
        ("cat", best_cat)
    ]
    stack = StackingRegressor(
        estimators=estimators_stack,
        final_estimator=vr,
        n_jobs=-1
    )
    stack.fit(X_train, y_train)
    y_pred_stack = stack.predict(X_val)
    rmse_stack = np.sqrt(mean_squared_error(y_val, y_pred_stack))

    results = {
        "lr": (lr, rmse_lr),
        "rfr": (best_rfr, rmse_rfr),
        "xgb": (best_xgb, rmse_xgb),
        "cat": (best_cat, rmse_cat),
        "vr": (vr, rmse_vr),
        "stack": (stack, rmse_stack)
    }
    return results

models = train_ensemble_model(train_df, preprocessor)

st.subheader("Primerjava RMSE (testna množica)")
rmse_dict = {name: round(rmse, 4) for name, (_, rmse) in models.items()}
rmse_df = pd.DataFrame.from_dict(rmse_dict, orient="index", columns=["RMSE"])
st.table(rmse_df)

st.sidebar.header("Predikcija na novih podatkih")
uploaded_file = st.sidebar.file_uploader(
    "Naloži CSV datoteko s stolpci, enakimi kot v končnem train_df (brez SalePrice)",
    type=["csv"]
)

if uploaded_file is not None:
    new_data = pd.read_csv(uploaded_file)

    required_cols = list(train_df.drop("SalePrice", axis=1).columns)
    missing = set(required_cols) - set(new_data.columns)
    if missing:
        st.sidebar.error(f"Manjkajoči stolpci: {sorted(missing)}")
    else:

        new_preproc = preprocessor.transform(new_data[required_cols])
        best_model = models["stack"][0]
        y_logpred = best_model.predict(new_preproc)
        y_pred = np.expm1(y_logpred)
        output_df = new_data.copy()
        output_df["PredictedSalePrice"] = y_pred
        st.subheader("Rezultati napovedi")
        st.dataframe(output_df)

        csv = output_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Prenesi napovedi kot CSV",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )

if st.checkbox("Prikaži korelacijsko matriko (numerične spremenljivke)"):
    corr = train_df.select_dtypes(include=["int64", "float64"]).corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)


if st.checkbox("Pomembnost značilk (RandomForest)"):
    rfr_model = models["rfr"][0]
    fi = rfr_model.feature_importances_
    feat_names = train_df.drop("SalePrice", axis=1).columns
    fi_df = pd.DataFrame({"Feature": feat_names, "Importance": fi})
    fi_df = fi_df.sort_values("Importance", ascending=False).head(20)

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.barplot(x="Importance", y="Feature", data=fi_df, palette="viridis", ax=ax2)
    ax2.set_title("Top 20 pomembnih značilk (RandomForest)")
    st.pyplot(fig2)
