import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix)
from sklearn.feature_selection import SelectKBest, f_classif


print("--- Завдання 2: Завантаження та первинний огляд ---")
df = pd.read_csv('covid_data.csv')

print("Перші 5 рядків:")
print(df.head())
print("\nІнформація про датасет:")
df.info()

print("\n--- Завдання 3: Очищення даних ---")
print("Кількість відсутніх значень до очищення:\n", df[['new_cases', 'new_deaths', 'total_cases']].isnull().sum())

df = df.drop_duplicates()
df['date'] = pd.to_datetime(df['date'])

df = df.sort_values(by=['location', 'date'])
df['new_cases'] = df['new_cases'].fillna(0)
df['new_deaths'] = df['new_deaths'].fillna(0)
df['total_cases'] = df.groupby('location')['total_cases'].transform(lambda x: x.ffill().fillna(0))
df = df.drop_duplicates()

print("\n--- Завдання 4: Статистичний огляд ---")
key_metrics = ['total_cases', 'new_cases', 'total_deaths', 'new_deaths']
print(df[key_metrics].describe())

print("\n--- Завдання 5: Візуалізація ---")
df_ukraine = df[df['location'] == 'Ukraine'].sort_values('date')
plt.figure(figsize=(12, 6))
plt.plot(df_ukraine['date'], df_ukraine['total_cases'], label='Загальна к-ть випадків', color='blue')
plt.plot(df_ukraine['date'], df_ukraine['total_deaths'], label='Загальна к-ть смертей', color='red')
plt.title('Динаміка випадків COVID-19 та смертей в Україні')
plt.xlabel('Дата')
plt.ylabel('Кількість')
plt.legend()
plt.grid(True)
plt.savefig('01_ukraine_dynamics.png')
plt.close()

df_latest_correct = df.groupby(['continent', 'location'])['total_cases'].max().reset_index()
continent_cases = df_latest_correct.groupby('continent')['total_cases'].sum().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=continent_cases.index, y=continent_cases.values, hue=continent_cases.index, palette='viridis',
            legend=False)
plt.title('Загальна кількість випадків COVID-19 по континентах')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('02_continents_cases.png')
plt.close()

print("\n--- Завдання 7: Трансформація даних ---")
df['location_encoded'] = df['iso_code']
df['growth_rate_new_cases'] = df.groupby('location')['new_cases'].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
df['growth_rate_new_deaths'] = df.groupby('location')['new_deaths'].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
df.to_csv('covid_data_cleaned.csv', index=False)
print("Дані збережено у 'covid_data_cleaned.csv'.")

print("\n--- Завдання 8: Кореляційний аналіз ---")
corr_cols = ['new_cases', 'new_deaths', 'total_cases', 'population', 'gdp_per_capita']
corr_matrix = df[corr_cols].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Кореляційна матриця')
plt.tight_layout()
plt.savefig('03_correlation_matrix.png')
plt.close()

print("\n--- Завдання 9: Розподіл даних ---")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(df['total_cases'], bins=50, ax=axes[0], color='blue', log_scale=(False, True))
axes[0].set_title('Розподіл загальної кількості випадків (Log Scale)')
sns.histplot(df['total_deaths'].dropna(), bins=50, ax=axes[1], color='red', log_scale=(False, True))
axes[1].set_title('Розподіл смертей (Log Scale)')
plt.tight_layout()
plt.savefig('04_histograms.png')
plt.close()

plt.figure(figsize=(10, 6))
sns.boxplot(x='continent', y='total_deaths_per_million', data=df[df['continent'].notna()])
plt.title('Розподіл смертей на мільйон за континентами')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('05_boxplot.png')
plt.close()

pairplot_cols = ['total_cases', 'total_deaths', 'total_vaccinations', 'population']
sns.pairplot(df[pairplot_cols].dropna().sample(n=1000, random_state=42), diag_kind='kde')
plt.suptitle('Pairplot для ключових змінних', y=1.02)
plt.savefig('06_pairplot.png')
plt.close()

print("\n--- Завдання 10: Аналіз трендів ---")
target_countries = ['United States', 'Ukraine', 'Germany']
df_targets = df[df['location'].isin(target_countries)]
fig, axes = plt.subplots(2, 1, figsize=(12, 10))
for country in target_countries:
    c_data = df_targets[df_targets['location'] == country]
    axes[0].plot(c_data['date'], c_data['new_cases'], label=country)
    axes[1].plot(c_data['date'], c_data['new_deaths'], label=country, linestyle='--')
axes[0].set_title('Тренд нових випадків')
axes[0].legend()
axes[1].set_title('Тренд нових смертей')
axes[1].legend()
plt.tight_layout()
plt.savefig('07_trends.png')
plt.close()

df_latest_per_country = df[df['continent'].notna()].groupby('location').last().reset_index()
top_10 = df_latest_per_country.nlargest(10, 'total_cases_per_million')
plt.figure(figsize=(10, 6))
sns.barplot(x='total_cases_per_million', y='location', data=top_10, hue='location', palette='magma', legend=False)
plt.title('Топ-10 країн за кількістю випадків на мільйон')
plt.tight_layout()
plt.savefig('08_top10_countries.png')
plt.close()

print("Перший блок візуалізацій успішно збережено!")

print("\n--- Завантаження та підготовка даних для ML ---")
df = pd.read_csv('covid_clean.csv')
df = df.sample(n=25000, random_state=42).reset_index(drop=True)
df = df.fillna(0)

df['high_cases'] = (df['new_cases'] > 1000).astype(int)

if 'continent' in df.columns:
    df = pd.get_dummies(df, columns=['continent'], drop_first=True)

le = LabelEncoder()
df['iso_code'] = le.fit_transform(df['iso_code'].astype(str))

cols_to_drop = ['new_cases', 'high_cases', 'date', 'location', 'new_cases_smoothed', 'new_cases_per_million',
                'new_cases_smoothed_per_million']
X = df.drop(columns=[c for c in cols_to_drop if c in df.columns]).select_dtypes(include=[np.number])

scaler = StandardScaler()
cols_to_scale = ['total_cases', 'total_deaths', 'total_vaccinations']
for col in cols_to_scale:
    if col in X.columns:
        X[col] = scaler.fit_transform(X[[col]])

y_reg = df['new_cases']
y_clf = df['high_cases']

Xr_train, Xr_test, yr_train, yr_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
Xc_train, Xc_test, yc_train, yc_test = train_test_split(X, y_clf, test_size=0.2, random_state=42)

smote = SMOTE(random_state=42)
Xc_train_sm, yc_train_sm = smote.fit_resample(Xc_train, yc_train)

print("\n--- Оцінка регресійних моделей ---")
reg_models = {
    'Linear': LinearRegression(),
    'Polynomial(deg=2)': make_pipeline(PolynomialFeatures(degree=2, include_bias=False), Ridge()),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1)
}

best_reg_model = None
best_rmse = float('inf')
reg_results = []

for name, model in reg_models.items():
    model.fit(Xr_train, yr_train)
    y_pred = model.predict(Xr_test)
    mse = mean_squared_error(yr_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(yr_test, y_pred)
    r2 = r2_score(yr_test, y_pred)

    reg_results.append({'Model': name, 'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2})
    if rmse < best_rmse:
        best_rmse = rmse
        best_reg_model = model

print(pd.DataFrame(reg_results).to_string(index=False))

print("\n--- Оцінка класифікаційних моделей ---")
clf_models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42),
    'k-NN': KNeighborsClassifier(n_neighbors=5)
}

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()
min_fn = float('inf')
best_fn_model = ""

for i, (name, model) in enumerate(clf_models.items()):
    model.fit(Xc_train_sm, yc_train_sm)
    y_pred = model.predict(Xc_test)

    acc, prec, rec, f1 = accuracy_score(yc_test, y_pred), precision_score(yc_test, y_pred), recall_score(yc_test,
                                                                                                         y_pred), f1_score(
        yc_test, y_pred)
    cm = confusion_matrix(yc_test, y_pred)
    fn = cm[1][0]

    if fn < min_fn:
        min_fn = fn
        best_fn_model = name

    print(f"{name:<20} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | FN: {fn}")
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=axes[i])
    axes[i].set_title(f'{name} (FN={fn})')

plt.tight_layout()
plt.savefig('09_ml_confusion_matrices.png')
plt.close()
print(f"Найменше пропущених спалахів (FN) має: {best_fn_model}")

print("\n--- Завдання 13: Відбір ознак (SelectKBest) ---")
selector = SelectKBest(score_func=f_classif, k=10)
Xc_train_sel = selector.fit_transform(Xc_train_sm, yc_train_sm)
best_features = X.columns[selector.get_support()].tolist()
print("Топ-10 ознак:", best_features)

print("\n--- Крос-валідація (5-Fold) ---")
for name, model in clf_models.items():
    scores = cross_val_score(model, Xc_train_sm, yc_train_sm, cv=5, scoring='f1', n_jobs=-1)
    print(f"{name:<20} | Середній F1: {scores.mean():.4f} | Std: {scores.std():.4f}")

print("\n--- Оптимізація та Gradient Boosting ---")
param_grid = {'n_estimators': [50, 100], 'max_depth': [10, 20], 'min_samples_split': [2, 5]}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='f1', n_jobs=-1)
rf_grid.fit(Xc_train_sm, yc_train_sm)
print(f"Найкращі параметри RF: {rf_grid.best_params_}")
best_rf = rf_grid.best_estimator_

gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(Xc_train_sm, yc_train_sm)
y_pred_gb = gb_model.predict(Xc_test)
print(f"Gradient Boosting | Acc: {accuracy_score(yc_test, y_pred_gb):.4f} | F1: {f1_score(yc_test, y_pred_gb):.4f}")

print("\n--- Фінальне прогнозування ---")
y_pred_final_reg = best_reg_model.predict(Xr_test)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].scatter(yr_test, y_pred_final_reg, alpha=0.3, color='blue')
axes[0].plot([yr_test.min(), yr_test.max()], [yr_test.min(), yr_test.max()], 'r--', lw=2)
axes[0].set_title('Predicted vs Actual (Регресія)')
axes[0].set_xlabel('Фактичні випадки')
axes[0].set_ylabel('Передбачені випадки')

sns.histplot(yr_test - y_pred_final_reg, bins=50, kde=True, ax=axes[1], color='purple')
axes[1].set_title('Розподіл помилок (Residuals Histogram)')

plt.tight_layout()
plt.savefig('10_regression_analysis.png')
plt.close()