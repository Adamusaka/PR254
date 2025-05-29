import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb


train_df = pd.read_csv("./Podatki/train.csv")
test_df = pd.read_csv("./Podatki/test.csv")

train_df.describe()


# # Čiščenje podatkov
# 
# Odstranjujem osamelce in nadomeščam null vrednosti (Večina v "No"). To pa zaradi tega, ker nekaj podatkov če ima vrednost Nan pomeni, da določene lastnosti nima, ne da nevemo če to sploh ima.



plt.scatter(x='LotFrontage', y='SalePrice', data=train_df)
# LotFrontage: Linear feet of street connected to property



train_df.query('LotFrontage > 300')
# Pregledal sem nekatere pomembne atribute kot LotArea, LotFrontage in izbrisal najdene osmalece
# 935, 1299, 250, 314, 336, 707, 
# morda: 1397, 

z_scores = pd.Series(stats.zscore(train_df['LotArea']))
z_scores.sort_values(ascending=False).head(10)
# iskanje z scora (za kok std je daleč od povprečja)

plt.scatter(x='OverallQual', y='SalePrice', data=train_df)
# Preštudiri še te osamelce npr za oceno 8un nad 500000 al pa 4 nad 200000


# we shold be dropping in the attributes that have a higher correlation
plt.scatter(x='OverallCond', y='SalePrice', data=train_df)
train_df.query('OverallCond == 6 & SalePrice > 700000')
# ta ni tok pomemben attrib
# 379, 1183, 692


plt.scatter(x='YearBuilt', y='SalePrice', data=train_df)
train_df.query('YearBuilt < 1900 & SalePrice > 400000')
#186


# Tukaj sem opravil še veliko več iskanja osamelcev
# Treba je tudi pasti, da ne kar odstranjuješ vse osamelce, 
# ker nekateri imajo lahko kakšne uporabne podatke

values = [598, 955, 935, 1299, 250, 314, 336, 707, 379, 1183, 692, 186, 441, 186, 524, 739, 598, 955, 636, 1062, 1191, 496, 198, 1338]

# Odstranimo vrednosti
train_df = train_df[train_df.Id.isin(values) == False]

# Zopet pregled null vrednosti
train_df_arranged = pd.DataFrame(train_df.isnull().sum().sort_values(ascending=False))
train_df_arranged.head(20)
# Te lahko nadomestimo s povp ali jih odstranimo

objectAtrib = ['FireplaceQu', 'MasVnrType', 'Fence', 'Alley', 'GarageCond', 'GarageFinish', 'GarageQual', 'BsmtExposure', 'BsmtQual', 'BsmtCond']
# BsmtExposure probi mal ekspermientirat
intAtrib = ['MasVnrArea', 'LotFrontage']
unfAtrib = ['BsmtFinType1', 'BsmtFinType2']

for atrib in objectAtrib:
    train_df[atrib].fillna('No', inplace=True)
    test_df[atrib].fillna('No', inplace=True)

for atrib in intAtrib:
    train_df[atrib].fillna(0, inplace=True)
    test_df[atrib].fillna(0, inplace=True)

for atrib in unfAtrib:
    # Unf - unfinished
    train_df[atrib].fillna('Unf', inplace=True)
    test_df[atrib].fillna('Unf', inplace=True)

# SBrkr	Standard Circuit Breakers & Romex
train_df['Electrical'].fillna('SBrkr', inplace=True)
test_df['Electrical'].fillna('SBrkr', inplace=True)

#Blok kode samo za testiranje
sns.catplot(data=train_df, x="GarageType", y="SalePrice", kind="box")
print(train_df['GarageType'].unique())
# TA -typicall


train_df = train_df.drop(columns=['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'GarageYrBlt', 'GarageCond', 'BsmtFinType2'])
test_df = test_df.drop(columns=['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'GarageYrBlt', 'GarageCond', 'BsmtFinType2'])

# Probi še brez droppanja Fence, Alley, pa garage zadev, BsmFinType


# # Feature engineering
# 
# Gradil bom različne funkcije, zgradi več funkcij kokr pri testiranju zato, probaš optim čim večji score. Prve črke na malo zanalašč, da se lahko prepozna na novo dodane.

train_df['houseAge'] = train_df['YrSold'] - train_df['YearBuilt']
test_df['houseAge'] = test_df['YrSold'] - test_df['YearBuilt']


train_df['houseRemodelAge'] = train_df['YrSold'] - train_df['YearRemodAdd']
test_df['houseRemodelAge'] = test_df['YrSold'] - test_df['YearRemodAdd']

train_df['totalSF'] = train_df['1stFlrSF'] + train_df['2ndFlrSF'] + train_df['BsmtFinSF1'] + train_df['BsmtFinSF2']
test_df['totalSF'] = test_df['1stFlrSF'] + test_df['2ndFlrSF'] + test_df['BsmtFinSF1'] + test_df['BsmtFinSF2']
# square feet

train_df['totalArea'] = train_df['GrLivArea'] + train_df['TotalBsmtSF']
test_df['totalArea'] = test_df['GrLivArea'] + test_df['TotalBsmtSF']

train_df['totalBaths'] = train_df['BsmtFullBath'] + train_df['FullBath'] + 0.5 * (train_df['BsmtHalfBath'] + train_df['HalfBath']) 
test_df['totalBaths'] = test_df['BsmtFullBath'] + test_df['FullBath'] + 0.5 * (test_df['BsmtHalfBath'] + test_df['HalfBath']) 

train_df['totalPorchSF'] = train_df['OpenPorchSF'] + train_df['3SsnPorch'] + train_df['EnclosedPorch'] + train_df['ScreenPorch'] + train_df['WoodDeckSF']
test_df['totalPorchSF'] = test_df['OpenPorchSF'] + test_df['3SsnPorch'] + test_df['EnclosedPorch'] + test_df['ScreenPorch'] + test_df['WoodDeckSF']

train_df = train_df.drop(columns=['Id','YrSold', 'YearBuilt', 'YearRemodAdd', '1stFlrSF', '2ndFlrSF', 'BsmtFinSF1', 'BsmtFinSF2', 'GrLivArea', 'TotalBsmtSF','BsmtFullBath', 'FullBath', 'BsmtHalfBath', 'HalfBath', 'OpenPorchSF', '3SsnPorch', 'EnclosedPorch', 'ScreenPorch','WoodDeckSF'])
test_df = test_df.drop(columns=['YrSold', 'YearBuilt', 'YearRemodAdd', '1stFlrSF', '2ndFlrSF', 'BsmtFinSF1', 'BsmtFinSF2', 'GrLivArea', 'TotalBsmtSF','BsmtFullBath', 'FullBath', 'BsmtHalfBath', 'HalfBath', 'OpenPorchSF', '3SsnPorch', 'EnclosedPorch', 'ScreenPorch','WoodDeckSF'])


# korelacija nvega 
correlation_matrix = train_df.corr(numeric_only=True)
plt.figure(figsize=(20,12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")


# GarageArea ali GarageCars TO ŠE STESTIRAJ pazi GrageArea ima neki outlierjov (vsaj zgledajo)
train_df = train_df.drop(columns=['GarageArea'])
test_df = test_df.drop(columns=['GarageArea'])

sns.histplot(train_df, x=train_df['SalePrice'])
# asimetrično

# V prejšnem zagovoru smo rekli, da bomo naredili log transformacijo nad SalePrice, 
# to je transformacija, ki se znebi asimetrije v atributu in zgleda veliko bolje normalno porazdeljena
# log1p = ln(1 + x)

train_df['SalePrice'] = np.log1p(train_df['SalePrice'])
sns.histplot(train_df, x=train_df['SalePrice'])

train_df.dtypes[train_df.dtypes=='object']
# Zdaj bomo rabli spremeniti vse te atribute, ker nemorš v modelih uporabljati kategoričnih spremenljivk

train_df.dtypes[train_df.dtypes!='object']

# Ordinal - Vrstni red je pomemben
# Ordinal bo naredil samo en column in začne od 0 in povešuje kasneje
ode_cols = ['LotShape', 'LandContour','Utilities','LandSlope',  'BsmtQual',  'BsmtFinType1',  'CentralAir',  'Functional', \
'FireplaceQu', 'GarageFinish', 'GarageQual', 'PavedDrive', 'ExterCond', 'KitchenQual', 'BsmtExposure', 'HeatingQC','ExterQual', 'BsmtCond']

# One hot - Vrstni red ni pomemen
# OneHotEndcoding bo naredil nove columne za vsakega (pretvori v svojo binarni stolpec: 1 če je vrednost in 0 če ni)
# Dobr za linearne modele sam 
ohe_cols = ['Street', 'LotConfig','Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'Exterior1st', 'Exterior2nd', \
           'MasVnrType','Foundation',  'Electrical',  'SaleType', 'MSZoning', 'SaleCondition', 'Heating', 'GarageType', 'RoofMatl']

num_cols = train_df.select_dtypes(include=['int64', 'float64']).columns
num_cols = num_cols.drop('SalePrice')

# Ustvarjam pipeline, da pretvorimo podatke
num_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
])

ode_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('ode', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

ohe_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

col_trans = ColumnTransformer(transformers=[
    ('num_p', num_pipeline, num_cols),
    ('ode_p', ode_pipeline, ode_cols),
    ('ohe_p', ohe_pipeline, ohe_cols),
    ],
    remainder='passthrough', 
    n_jobs=-1)

pipeline = Pipeline(steps=[
    ('preprocessing', col_trans)
])

X = train_df.drop('SalePrice', axis=1)
y = train_df['SalePrice']

X_preprocessed = pipeline.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=25)


# # Grajenje modelov

lr = LinearRegression()

lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

mean_squared_error(y_test, y_pred_lr)
# bl slaba

RFR = RandomForestRegressor(random_state=13)


param_grid_RFR = {
    'max_depth': [5, 10, 15],
    'n_estimators': [100, 250, 500],
    'min_samples_split': [3, 5, 10]
}

rfr_cv = GridSearchCV(RFR, param_grid_RFR, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)


rfr_cv.fit(X_train, y_train)

np.sqrt(-1 * rfr_cv.best_score_)

rfr_cv.best_params_

XGB = XGBRegressor(random_state=13)

param_grid_XGB = {
    'learning_rate': [0.05, 0.1, 0.2],
    'n_estimators': [300],
    'max_depth': [3],
    'min_child_weight': [1,2,3],
    'gamma': [0, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
}

xgb_cv = GridSearchCV(XGB, param_grid_XGB, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)

xgb_cv.fit(X_train, y_train)

np.sqrt(-1 * xgb_cv.best_score_)

ridge = Ridge()

param_grid_ridge = {
    'alpha': [0.05, 0.1, 1, 3, 5, 10],
    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag']
}

ridge_cv = GridSearchCV(ridge, param_grid_ridge, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

ridge_cv.fit(X_train, y_train)

np.sqrt(-1 * ridge_cv.best_score_)

GBR = GradientBoostingRegressor()

param_grid_GBR = {
    'max_depth': [12, 15, 20],
    'n_estimators': [200, 300, 1000],
    'min_samples_leaf': [10, 25, 50],
    'learning_rate': [0.001, 0.01, 0.1],
    'max_features': [0.01, 0.1, 0.7]
}

GBR_cv = GridSearchCV(GBR, param_grid_GBR, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

GBR_cv.fit(X_train, y_train)


np.sqrt(-1 * GBR_cv.best_score_)

# Tukaj je pomojem napaka z eno one hot spremenljivko

# lgbm_regressor = lgb.LGBMRegressor()

# param_grid_lgbm = {
#     'boosting_type': ['gbdt', 'dart'],
#     'num_leaves': [20, 30, 40],
#     'learning_rate': [0.01, 0.05, 0.1],
#     'n_estimators': [100, 200, 300]
# }

#lgbm_cv = GridSearchCV(lgbm_regressor, param_grid_lgbm, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)


# Morda napaka z neko onehot spremenljivko
# lgbm_cv.fit(X_train, y_train)


# np.sqrt(-1 * lgbm_cv.best_score_)

catboost = CatBoostRegressor(loss_function='RMSE', verbose=False)


param_grid_cat ={
    'iterations': [100, 500, 1000],
    'depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.5]
}

cat_cv = GridSearchCV(catboost, param_grid_cat, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)

cat_cv.fit(X_train, y_train)


np.sqrt(-1 * cat_cv.best_score_)

vr = VotingRegressor([('gbr', GBR_cv.best_estimator_),
                      ('xgb', xgb_cv.best_estimator_),
                      ('ridge', ridge_cv.best_estimator_)],
                    weights=[2,3,1])

vr.fit(X_train, y_train)

y_pred_vr = vr.predict(X_test)

mean_squared_error(y_test, y_pred_vr)

estimators = [
    ('gbr', GBR_cv.best_estimator_),
    ('xgb', xgb_cv.best_estimator_),
    ('cat', cat_cv.best_estimator_),
    #('lgb', lgbm_cv.best_estimator_),
    ('rfr', rfr_cv.best_estimator_),
]

stackreg = StackingRegressor(
            estimators = estimators,
            final_estimator = vr
)

stackreg.fit(X_train, y_train)

y_pred_stack = stackreg.predict(X_test)

mean_squared_error(y_test, y_pred_stack)

df_test_preprocess = pipeline.transform(test_df)


y_stacking = np.exp(stackreg.predict(df_test_preprocess))

df_y_stacking_out = test_df[['Id']]
df_y_stacking_out['SalePrice'] = y_stacking

df_y_stacking_out.to_csv('submission.csv', index=False)

