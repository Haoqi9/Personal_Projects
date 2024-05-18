import joblib
import numpy as np
import pandas as pd

# Load train set (full data) for imputing missings.
df_train_full = pd.read_csv('../02_Data/Original/prestamos.csv')

# Crear la target en el conjunto original:
class_neg = ["Fully Paid", "Current", "In Grace Period", "Late (16-30 days)", "Does not meet the credit policy. Status:Fully Paid"]
class_pos = ["Charged Off", "Late (31-120 days)", "Does not meet the credit policy. Status:Charged Off", "Default"]
df_train_full['default'] = np.where(df_train_full['estado'].isin(class_pos), 1, 0)

# Load final variables:
vars_final = joblib.load('../06_Other/final_variables')

# load trained final model:
model_final = joblib.load('../04_Models/best_lgbmc_cw_trained_all_data')

# Preprocessing functions:
def impute_missing(df_train, X_test):
    # Evitar sobreescribir variable.
    df_train = df_train.copy()
    X_test = X_test.copy()

    # Eliminar registros del conjunto de entrenamiento.
    mask_zero_notVerified = (df_train.ingresos == 0) & (df_train.ingresos_verificados == 'Not Verified')
    df_train = df_train.loc[~mask_zero_notVerified] 
    
    mask_dbt_neg = df_train.dti < 0
    df_train = df_train[~mask_dbt_neg]    
    
    rare_cats = ["ANY", "OTHER", "NONE"]
    mask_vivienda_rare = df_train.vivienda.isin(rare_cats)
    df_train = df_train[~mask_vivienda_rare]

    # Partición X_train / y_train.
    X_train = df_train.drop(columns='default')

    # Imputación por la moda.
    vars_moda = ["antigüedad_empleo", "num_hipotecas", "num_derogatorios"]  
    for var in vars_moda:
        X_test[var].fillna(X_train[var].mode().iloc[0], inplace=True)

    # Imputación por la mediana.
    vars_mediana = ["porc_uso_revolving", "dti", "num_lineas_credito", "porc_tarjetas_75p"]
    for var in vars_mediana:
        X_test[var].fillna(X_train[var].median(), inplace=True)

    return (X_train, X_test)
    

def get_derived_features(X):
    # Evitar sobreescribir variable.
    X = X.copy()

    # Agrupar categorías infrecuentes.
    X["rating"].replace(
        to_replace=["F", "G"],
        value="F-G",
        inplace=True
    )
    
    X["finalidad"].replace(
        to_replace=["major_purchase", "medical", "small_business", "car", "moving", "vacation", "house", "wedding", "renewable_energy", "educational"],
        value="other",
        inplace=True
    )
    
    # OHE.
    X = pd.get_dummies(
        data=X,
        columns=['antigüedad_empleo', 'finalidad', 'ingresos_verificados', 'rating', 'vivienda'],
        drop_first=False,
        dtype=float
    )

    return X


def preprocess_data(df_train, X_test, vars_final):
    # Evitar sobreescribir.
    df_train = df_train.copy()
    X_test  = X_test.copy()

    # Imputar nulos.
    X_train, X_test = impute_missing(df_train, X_test)

    # Feature engineering.
    X_test = get_derived_features(X_test)

    # Cambiar nombre de categorías acorde al output generado por LGBM.
    X_test.rename(columns={
        'num_cuotas_ 36 months': 'num_cuotas__36_months',
        'antigüedad_empleo_10+ years': 'antigüedad_empleo_10+_years',
        'antigüedad_empleo_2 years': 'antigüedad_empleo_2_years',
        'antigüedad_empleo_3 years': 'antigüedad_empleo_3_years',
        'ingresos_verificados_Not Verified': 'ingresos_verificados_Not_Verified',
        'ingresos_verificados_Source Verified': 'ingresos_verificados_Source_Verified'
    }, inplace=True)
    
    # Limitar a las variables finales.
    X_test_prep  = X_test[vars_final]

    return X_test_prep

# Batch production prediction function:
def predictProba_X_test(df_train_full, X_test, vars_final, model_final):
    X_test_prep = preprocess_data(df_train_full, X_test, vars_final)
    array_pred_default_prob = model_final.predict_proba(X_test_prep)[:, 1]
    
    return array_pred_default_prob