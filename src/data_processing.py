# `data_processing.py`: código para procesar los datos de la carpeta `data/raw` y guardar los datos procesados en la carpeta `data/processed`.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

lc = pd.read_csv('../data/raw/lending_club_loan_two.csv')

lc['issue_d'] = pd.to_datetime(lc['issue_d'], format='%b-%Y')

lc_2012 = lc[(lc['issue_d'] >= '2012-01-01') & (lc['issue_d'] <= '2012-12-31')]

lc_2012.to_csv('../data/processed/lc_2012.csv', index=False)

lc_2012['loan_status'] = lc_2012.loan_status.map({'Fully Paid':1, 'Charged Off':0})

term_values = {' 36 months': 36, ' 60 months': 60}
lc_2012['term'] = lc_2012.term.map(term_values)

# Mapa de reemplazo
emp_length_map = {
    '10+ years': 10,
    '9 years': 9,
    '8 years': 8,
    '7 years': 7,
    '6 years': 6,
    '5 years': 5,
    '4 years': 4,
    '3 years': 3,
    '2 years': 2,
    '1 year': 1,
    '< 1 year': 0
}

# Reemplazar los valores en la columna 'emp_length'
lc_2012['emp_length'] = lc_2012['emp_length'].replace(emp_length_map)

# Calcular la media de la columna 'emp_length' (excluyendo NaN)
media_emp_length = lc_2012['emp_length'].mean()

# Reemplazar NaN con la media de la columna
lc_2012['emp_length'] = lc_2012['emp_length'].fillna(media_emp_length)


lc_2012.drop('title', axis=1, inplace=True)

lc_2012 = lc_2012.drop(['verification_status','application_type'], axis=1)

# Infiero la mayor calidad crediticia a cada subgrade
# Mapeo de valores para la columna 'sub_grade'
sub_grade_map = {
    'A1': 35, 'A2': 34, 'A3': 33, 'A4': 32, 'A5': 31,
    'B1': 30, 'B2': 29, 'B3': 28, 'B4': 27, 'B5': 26,
    'C1': 25, 'C2': 24, 'C3': 23, 'C4': 22, 'C5': 21,
    'D1': 20, 'D2': 19, 'D3': 18, 'D4': 17, 'D5': 16,
    'E1': 15, 'E2': 14, 'E3': 13, 'E4': 12, 'E5': 11,
    'F1': 10, 'F2': 9, 'F3': 8, 'F4': 7, 'F5': 6,
    'G1': 5, 'G2': 4, 'G3': 3, 'G4': 2, 'G5': 1
}

# Reemplazar los valores en la columna 'sub_grade'
lc_2012['sub_grade'] = lc_2012['sub_grade'].replace(sub_grade_map)

lc_2012.drop('grade', axis=1, inplace=True)

# Calcular la media de la columna 'mort_acc' (excluyendo NaN)
media_mort_acc = lc_2012['mort_acc'].mean()

# Reemplazar NaN con la media de la columna
lc_2012['mort_acc'] = lc_2012['mort_acc'].fillna(media_mort_acc)


# revol_util  tiene..........41169 non-null  float64, o sea muy pocos null, los elimino
lc_2012 = lc_2012.dropna(subset=['revol_util'])

# No considero address determinante para este estudio, de momento
# La linea mas temprana en el historial crediticio del usuario, tampoco le doy valor,podria compararlo con la edad, si la tuviera
# issue_d la fecha de emision no es ponderable en si se impaga o no, a no ser que tuvieramos en cuenta ciclos económicos.
# por ello dropeo todas esas variables

lc_2012 = lc_2012.drop(['address','earliest_cr_line','issue_d'], axis=1)

'''loan_amnt: Monto del préstamo solicitado.
int_rate: Tasa de interés del préstamo.
installment: Pago mensual del préstamo.
sub_grade: Subgrado asignado al préstamo por LendingClub (A1, A2, B1, etc.).
open_acc: Número de líneas de crédito abiertas en el archivo del prestatario.
pub_rec: Número de registros públicos desfavorables.
total_acc: Número total de cuentas de crédito del prestatario.
pub_rec_bankruptcies: Número de quiebras en los registros públicos.'''

# El interes y la subcategorizacion estan relacionados en un -98%, a mayor categoria menor tipo, quito subcategoria
# El capital solicitado tiene un 96 % de corr con la cuota, normal la cuota es segun el capital, quito capital
# Las cuenta abiertas y totales estan relacionadas en un 66%, normal las abiertas estan en las totales, quito las abiertas
# Las quiebras y los resgistros desfavorables se relacionan en un89%, normal las quiebras son desfavorables, quito las quiebras,
# Lo repense y no lo quite porque queria inferir en los modelos la relación de estar en quiebra y su influencia en la target, riesgo de impago

lc_2012 = lc_2012.drop(['sub_grade','loan_amnt','open_acc'], axis=1)

# OJO PARA BALANCEAR LA MUESTRA, al estar las instancias en mas del 80% dando como resultado en la target, 1 Fully Paid.

from sklearn.utils import resample


lc_2012_majority = lc_2012[lc_2012['loan_status'] == 1]
lc_2012_minority = lc_2012[lc_2012['loan_status'] == 0]

lc_2012_majority_downsampled = resample(lc_2012_majority, replace=False, n_samples=len(lc_2012_minority), random_state=42)

lc_2012_balanced = pd.concat([lc_2012_majority_downsampled, lc_2012_minority])

lc_2012_balanced = lc_2012_balanced.drop(['emp_title'],axis=1)

lc_2012_balmix = lc_2012_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

medias_pub_rec_bankruptcies = lc_2012_balmix.groupby('pub_rec_bankruptcies')['loan_status'].mean()

# Asignar las medias de la target a las filas correspondientes
lc_2012_balmix['pub_rec_bankruptcies'] = lc_2012_balmix['pub_rec_bankruptcies'].map(medias_pub_rec_bankruptcies)

medias_mort_acc = lc_2012_balmix.groupby('mort_acc')['loan_status'].mean()

# Asignar las medias de target a las filas correspondientes
lc_2012_balmix['mort_acc'] = lc_2012_balmix['mort_acc'].map(medias_mort_acc)


medias_initial_list_status = lc_2012_balmix.groupby('initial_list_status')['loan_status'].mean()

# Asignar las medias de la target a las filas correspondientes
lc_2012_balmix['initial_list_status'] = lc_2012_balmix['initial_list_status'].map(medias_initial_list_status)


medias_total_acc = lc_2012_balmix.groupby('total_acc')['loan_status'].mean()

# Asignar las medias de la target a las filas correspondientes
lc_2012_balmix['total_acc'] = lc_2012_balmix['total_acc'].map(medias_total_acc)


medias_pub_rec = lc_2012_balmix.groupby('pub_rec')['loan_status'].mean()

# Asignar las medias de la target a las filas correspondientes
lc_2012_balmix['pub_rec'] = lc_2012_balmix['pub_rec'].map(medias_pub_rec)


medias_purpose = lc_2012_balmix.groupby('purpose')['loan_status'].mean()

# Asignar las medias de la target a las filas correspondientes
lc_2012_balmix['purpose'] = lc_2012_balmix['purpose'].map(medias_purpose)



medias_emp_length = lc_2012_balmix.groupby('emp_length')['loan_status'].mean()

# Asignar las medias de la target a las filas correspondientes
lc_2012_balmix['emp_length'] = lc_2012_balmix['emp_length'].map(medias_emp_length)


medias_home_ownership = lc_2012_balmix.groupby('home_ownership')['loan_status'].mean()

# Asignar las medias de la target a las filas correspondientes
lc_2012_balmix['home_ownership'] = lc_2012_balmix['home_ownership'].map(medias_home_ownership)


medias_term = lc_2012_balmix.groupby('term')['loan_status'].mean()

# Asignar las medias de la target a las filas correspondientes
lc_2012_balmix['term'] = lc_2012_balmix['term'].map(medias_term)


X_balmix_cat_mean= lc_2012_balmix.drop(['loan_status'], axis=1)

y_balmix_cat_mean= lc_2012_balmix.drop(['term','int_rate','installment','emp_length','home_ownership','annual_inc','purpose','dti','pub_rec','revol_bal','revol_util','total_acc','initial_list_status','mort_acc','pub_rec_bankruptcies'], axis=1)


X_balmix_cat_mean.to_csv('../data/processed/X_balmix_cat_mean.csv', index=False)
y_balmix_cat_mean.to_csv('../data/processed/y_balmix_cat_mean.csv', index=False)