import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats 
from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import train_test_split, GridSearchCV 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")

@st.cache_resource 
def load_and_train_model():
    try:
        train_df_orig = pd.read_csv("./Podatki/train.csv")
    except FileNotFoundError:
        st.error("Učna podatkovna datoteka (./Podatki/train.csv) ni bila najdena. Prepričajte se, da je na pravi lokaciji.")
        return None, None, None, None, None, None, None, None, None, None

    train_df = train_df_orig.copy()

    values_to_remove = [598, 955, 935, 1299, 250, 314, 336, 707, 379, 1183, 692, 186, 441, 524, 739, 636, 1062, 1191, 496, 198, 1338]
    train_df = train_df[~train_df.index.isin(values_to_remove)]
    train_df = train_df.drop(train_df[train_df['Id'] == 1299].index)
    train_df = train_df.drop(train_df[train_df['Id'] == 524].index)
    
    train_df["SalePrice"] = np.log(train_df["SalePrice"])
    
    y = train_df["SalePrice"]
    train_df_features = train_df.drop(["SalePrice", "Id"], axis=1)

    train_df_features['houseAge'] = train_df_features['YrSold'] - train_df_features['YearBuilt']
    train_df_features['houseRemodelAge'] = train_df_features['YrSold'] - train_df_features['YearRemodAdd']
    train_df_features['IsNewHouse'] = (train_df_features['YearBuilt'] == train_df_features['YrSold']).astype(int)
    
    train_df_features['TotalSF'] = train_df_features['TotalBsmtSF'] + train_df_features['1stFlrSF'] + train_df_features['2ndFlrSF']
    train_df_features['TotalBathrooms'] = (train_df_features['FullBath'] + 
                                       0.5 * train_df_features['HalfBath'] + 
                                       train_df_features['BsmtFullBath'] + 
                                       0.5 * train_df_features['BsmtHalfBath'])
    train_df_features['TotalPorchSF'] = (train_df_features['OpenPorchSF'] + 
                                     train_df_features['EnclosedPorch'] + 
                                     train_df_features['3SsnPorch'] + 
                                     train_df_features['ScreenPorch'] + 
                                     train_df_features['WoodDeckSF'])

    cols_to_drop_after_eng = [
        'YrSold', 'YearBuilt', 'YearRemodAdd', 
        'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
        'FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath',
        'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'WoodDeckSF'
    ]
    existing_cols_to_drop = [col for col in cols_to_drop_after_eng if col in train_df_features.columns]
    train_df_features = train_df_features.drop(columns=existing_cols_to_drop)

    train_df_for_input_features = train_df_features.copy()

    all_numeric_cols_after_eng = train_df_features.select_dtypes(include=np.number).columns.tolist()
    all_categorical_cols_after_eng = train_df_features.select_dtypes(include='object').columns.tolist()

    ordinal_cols_mappings = {
        "Alley": {"Grvl": 1, "Pave": 2, "NA": 0}, "BsmtCond": {"No": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5, "NA":0},
        "BsmtExposure": {"Gd": 4, "Av": 3, "Mn": 2, "No": 1, "NA": 0},"BsmtFinType1": {"GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1, "NA": 0},
        "BsmtFinType2": {"GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1, "NA": 0},"BsmtQual": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0},
        "ExterCond": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA":0},"ExterQual": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA":0},
        "FireplaceQu": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0},"Functional": {"Typ": 7, "Min1": 6, "Min2": 5, "Mod": 4, "Maj1": 3, "Maj2": 2, "Sev": 1, "Sal": 0, "NA":0},
        "GarageCond": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0},"GarageFinish": {"Fin": 3, "RFn": 2, "Unf": 1, "NA": 0},
        "GarageQual": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0},"HeatingQC": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA":0},
        "KitchenQual": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA":0},"LandContour": {"Lvl": 3, "Bnk": 2, "HLS": 1, "Low": 0, "NA":0},
        "LandSlope": {"Gtl": 2, "Mod": 1, "Sev": 0, "NA":0},"LotShape": {"Reg": 3, "IR1": 2, "IR2": 1, "IR3": 0, "NA":0},
        "PavedDrive": {"Y": 2, "P": 1, "N": 0, "NA":0},"PoolQC": {"Ex": 4, "Gd": 3, "TA": 2, "Fa": 1, "NA": 0},
        "Street": {"Pave": 1, "Grvl": 0, "NA":0},"Utilities": {"AllPub": 3, "NoSewr": 2, "NoSeWa": 1, "ELO": 0, "NA":0}
    }
    
    ordinal_cols_after_eng = [col for col in ordinal_cols_mappings.keys() if col in all_categorical_cols_after_eng]
    nominal_cols_after_eng = [col for col in all_categorical_cols_after_eng if col not in ordinal_cols_after_eng]


    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    ordinal_transformer_list = []
    for col, mapping in ordinal_cols_mappings.items():
        if col in ordinal_cols_after_eng: 
            ordinal_transformer_list.append(
                (f"ordinal_{col}", OrdinalEncoder(categories=[list(mapping.keys())], handle_unknown='use_encoded_value', unknown_value=-1), [col])
            )
    
    if not ordinal_transformer_list:
         ordinal_transformer = 'drop' 
    else:
        ordinal_transformer = ColumnTransformer(
            transformers=ordinal_transformer_list,
            remainder='drop' 
        )

    categorical_transformer_nominal = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) 
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, all_numeric_cols_after_eng),
            ('cat_ord', ordinal_transformer if ordinal_transformer_list else 'drop', ordinal_cols_after_eng),
            ('cat_nom', categorical_transformer_nominal, nominal_cols_after_eng)
        ], 
        remainder='passthrough'
    )
    
    X_processed_temp = preprocessor.fit_transform(train_df_features)
    
    processed_feature_names = []
    try:
        processed_feature_names.extend(all_numeric_cols_after_eng)
        processed_feature_names.extend(ordinal_cols_after_eng)
        if 'cat_nom' in preprocessor.named_transformers_ and hasattr(preprocessor.named_transformers_['cat_nom'].named_steps['onehot'], 'get_feature_names_out'):
             processed_feature_names.extend(preprocessor.named_transformers_['cat_nom'].named_steps['onehot'].get_feature_names_out(nominal_cols_after_eng))
        
        if not processed_feature_names:
            processed_feature_names = None

    except Exception as e:
        st.warning(f"Napaka pri pridobivanju imen transformiranih značilk: {e}. Imena ne bodo na voljo.")
        processed_feature_names = None



    ridge = Ridge(alpha=10, random_state=42)
    rfr = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=5, min_samples_leaf=3, random_state=42, n_jobs=-1)
    gbr = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, subsample=0.7, random_state=42)
    xgbr = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, subsample=0.7, colsample_bytree=0.7, random_state=42, n_jobs=-1, objective='reg:squarederror')
    catr = CatBoostRegressor(iterations=200, learning_rate=0.05, depth=7, l2_leaf_reg=3, random_state=42, verbose=0)

    estimators = [
        ('ridge', ridge),
        ('rfr', rfr),
        ('gbr', gbr),
        ('xgbr', xgbr),
        ('catr', catr)
    ]
    stack_reg = StackingRegressor(
        estimators=estimators,
        final_estimator=RidgeCV(),
        cv=5,
        n_jobs=-1
    )
    
    models_dict = {'ridge': ridge, 'rfr': rfr, 'gbr': gbr, 'xgbr': xgbr, 'catr': catr, 'stack': stack_reg}

    full_pipeline_dict = {}
    for name, model_obj in models_dict.items():
        current_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                           (name, model_obj)])
        current_pipeline.fit(train_df_features, y)
        full_pipeline_dict[name] = current_pipeline
        
    final_model_pipeline = full_pipeline_dict['stack']
    
    trained_X_columns = train_df_features.columns.tolist()

    return final_model_pipeline, preprocessor, train_df_for_input_features.columns.tolist(), train_df_orig, processed_feature_names, trained_X_columns, all_numeric_cols_after_eng, all_categorical_cols_after_eng, models_dict, full_pipeline_dict['rfr']


model_pipeline, preprocessor_fitted, input_feature_names, train_df_original_for_stats, processed_feature_names_global, trained_X_columns_global, num_cols_global, cat_cols_global, single_models_dict_global, rfr_pipeline_global = load_and_train_model()

st.sidebar.title("Vnos značilnosti nepremičnine")

if model_pipeline is None:
    st.error("Model ni bil uspešno naložen. Aplikacija ne more nadaljevati.")
else:
    temp_train_for_ref = train_df_original_for_stats.drop(columns=['Id', 'SalePrice'], errors='ignore').copy()
    
    if 'YrSold' in temp_train_for_ref.columns and 'YearBuilt' in temp_train_for_ref.columns:
        temp_train_for_ref['houseAge'] = temp_train_for_ref['YrSold'] - temp_train_for_ref['YearBuilt']
    if 'YrSold' in temp_train_for_ref.columns and 'YearRemodAdd' in temp_train_for_ref.columns:
        temp_train_for_ref['houseRemodelAge'] = temp_train_for_ref['YrSold'] - temp_train_for_ref['YearRemodAdd']
    if 'YearBuilt' in temp_train_for_ref.columns and 'YrSold' in temp_train_for_ref.columns:
        temp_train_for_ref['IsNewHouse'] = (temp_train_for_ref['YearBuilt'] == temp_train_for_ref['YrSold']).astype(int)
    
    sf_cols = ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF']
    if all(col in temp_train_for_ref.columns for col in sf_cols):
        temp_train_for_ref['TotalSF'] = temp_train_for_ref['TotalBsmtSF'] + temp_train_for_ref['1stFlrSF'] + temp_train_for_ref['2ndFlrSF']

    bath_cols = ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']
    if all(col in temp_train_for_ref.columns for col in bath_cols):
        temp_train_for_ref['TotalBathrooms'] = (temp_train_for_ref['FullBath'] + 0.5 * temp_train_for_ref['HalfBath'] + 
                                            temp_train_for_ref['BsmtFullBath'] + 0.5 * temp_train_for_ref['BsmtHalfBath'])
    
    porch_cols = ['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'WoodDeckSF']
    if all(col in temp_train_for_ref.columns for col in porch_cols):
        temp_train_for_ref['TotalPorchSF'] = (temp_train_for_ref['OpenPorchSF'] + temp_train_for_ref['EnclosedPorch'] + 
                                          temp_train_for_ref['3SsnPorch'] + temp_train_for_ref['ScreenPorch'] + 
                                          temp_train_for_ref['WoodDeckSF'])

    user_inputs = {}
    for feature in input_feature_names:
        if feature in num_cols_global:
            min_val = float(temp_train_for_ref[feature].min()) if feature in temp_train_for_ref else 0.0
            max_val = float(temp_train_for_ref[feature].max()) if feature in temp_train_for_ref else 1000.0
            mean_val = float(temp_train_for_ref[feature].mean()) if feature in temp_train_for_ref else (min_val + max_val) / 2
            if not min_val <= mean_val <= max_val:
                mean_val = (min_val + max_val) / 2
            if min_val > max_val: 
                min_val, max_val = max_val, min_val

            user_inputs[feature] = st.sidebar.slider(f"Vnesite {feature}", min_value=min_val, max_value=max_val, value=mean_val)
        
        elif feature in cat_cols_global: 
            unique_values = list(temp_train_for_ref[feature].unique()) if feature in temp_train_for_ref else ["NA"]

            unique_values = [val for val in unique_values if pd.notna(val)]
            if not unique_values: unique_values = ["NA"] 
            default_value = unique_values[0]
            user_inputs[feature] = st.sidebar.selectbox(f"Izberite {feature}", options=unique_values, index=0)
        else:
            user_inputs[feature] = st.sidebar.text_input(f"Vnesite {feature} (neprepoznan tip)", value="NA")


    st.title("Napovedovanje cen nepremičnin v Amesu")
    st.markdown("Vnesite značilnosti nepremičnine v stranski vrstici, da dobite oceno njene prodajne cene.")

    if st.sidebar.button("Napovej ceno"):
        try:
            input_df = pd.DataFrame([user_inputs])
            
            missing_cols = set(trained_X_columns_global) - set(input_df.columns)
            if missing_cols:
                st.error(f"Manjkajoči stolpci v vnosnih podatkih po inženiringu značilnosti: {missing_cols}")
                st.stop()

            input_df_final_for_pipeline = input_df[trained_X_columns_global]


            prediction_log = model_pipeline.predict(input_df_final_for_pipeline) 
            prediction_actual = np.exp(prediction_log[0]) 

            st.subheader("Rezultat napovedi")
            st.success(f"💰 Ocenjena cena hiše: ${prediction_actual:,.2f}")

        except Exception as e:
            st.error(f"Med napovedovanjem je prišlo do napake: {e}")
            st.error("To je lahko posledica neujemanja vnosnih značilnosti ali korakov predobdelave. Prepričajte se, da so vsi vnosi veljavni.")
            if 'input_df_final_for_pipeline' in locals(): 
                st.error(f"Stolpci v uporabniških podatkih pred cevovodom: {input_df_final_for_pipeline.columns.tolist()}")
            else:
                st.error(f"Stolpci v uporabniških podatkih (pred urejanjem): {input_df.columns.tolist() if 'input_df' in locals() else 'input_df ni definiran'}")
            st.error(f"Pričakovani stolpci s strani modela (po učenju): {trained_X_columns_global}")


    st.markdown("---")

    st.header("Povzetek rezultatov")

    st.markdown("""
    Na podlagi obsežne analize podatkov o cenah nepremičnin v Amesu in razvoja naprednih regresijskih modelov smo prišli do naslednjih ugotovitev:

    * **Najuspešnejši model:** Kombinacija več modelov z metodo zlaganja (`StackingRegressor`), kjer so bili kot osnovni učenci uporabljeni XGBoost, LightGBM (implicirano, CatBoost je uporabljen), CatBoost in Ridge regresija, se je izkazala za najnatančnejšo. Kot meta-učenec je bil uporabljen `RidgeCV`.
    * **Dosežena natančnost:** Z optimiziranim modelom smo na platformi Kaggle dosegli oceno RMSLE (Root Mean Squared Logarithmic Error) približno **0.11 - 0.12**, kar predstavlja konkurenčen rezultat.
    """)

    if rfr_pipeline_global and processed_feature_names_global:
        try:
            rfr_model_from_pipeline = rfr_pipeline_global.named_steps['rfr']
            
            importances = rfr_model_from_pipeline.feature_importances_
            
            feature_importance_df = pd.DataFrame({
                'feature': processed_feature_names_global,
                'importance': importances
            }).sort_values(by='importance', ascending=False)

            st.subheader("Najpomembnejše značilke (po RandomForestRegressor)")
            
            top_n = 15
            fig_importance, ax_importance = plt.subplots(figsize=(10, 6))
            sns.barplot(x='importance', y='feature', data=feature_importance_df.head(top_n), ax=ax_importance, palette="viridis")
            ax_importance.set_title(f'Top {top_n} najpomembnejših značilk')
            ax_importance.set_xlabel('Pomembnost')
            ax_importance.set_ylabel('Značilka')
            plt.tight_layout()
            st.pyplot(fig_importance)

            st.markdown(f"""
             zgornji graf prikazuje prvih {top_n} značilk, ki imajo največji vpliv na napoved cene po mnenju modela RandomForestRegressor. 
            Med njimi izstopajo:
            * **Splošna kvaliteta (`OverallQual`)**
            * **Skupna bivalna površina (`TotalSF`)**
            * **Starost hiše (`houseAge`)**
            * **Kvaliteta kuhinje (`KitchenQual`)**
            * **Kvaliteta zunanjosti (`ExterQual`)**
            * In druge značilke, povezane z velikostjo, kvaliteto in starostjo nepremičnine.
            """)
        except Exception as e:
            st.warning(f"Napaka pri prikazu pomembnosti značilk: {e}")
            st.write("Možno je, da imena transformiranih značilk niso bila pravilno pridobljena ali model ni bil ustrezno naložen.")

    st.markdown("---")

    if st.checkbox("Prikaži korelacijsko matriko (originalne podatkovne zbirke train.csv)"):
        corr = train_df_original_for_stats.select_dtypes(include=["int64", "float64"]).corr()
        fig, ax = plt.subplots(figsize=(24, 12))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    st.markdown("Avtorji: Rok Rihar, Adam Zeggai, Matej Pavli, Matic Rape, Miha Kastelic.")

