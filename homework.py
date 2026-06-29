import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Завдання 1: Завершіть ETL-процес
print("Виконується Завдання 1: Очищення даних")
df = pd.read_csv('covid_data.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by=['location', 'date'])

fill_zero_cols = ['new_cases', 'new_deaths', 'new_cases_smoothed', 'new_deaths_smoothed', 'people_fully_vaccinated_per_hundred']
for col in fill_zero_cols:
    if col in df.columns:
        df[col] = df[col].fillna(0)

df['total_cases'] = df.groupby('location')['total_cases'].transform(lambda x: x.ffill().fillna(0))
df['total_deaths'] = df.groupby('location')['total_deaths'].transform(lambda x: x.ffill().fillna(0))

df = df.drop_duplicates()
df.to_csv('covid_clean.csv', index=False)
print("Дані очищено та збережено у 'covid_clean.csv'")


# Завдання 2: Розширений кореляційний аналіз
print("\nВиконується Завдання 2: Кореляційна матриця")
extended_cols = [
    'total_cases', 'total_deaths', 'new_cases', 'new_deaths',
    'population', 'gdp_per_capita', 'median_age', 'aged_65_older',
    'cardiovasc_death_rate', 'diabetes_prevalence',
    'hospital_beds_per_thousand', 'human_development_index'
]
corr_matrix = df[extended_cols].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='viridis', fmt=".2f", linewidths=0.5)
plt.title('Розширена кореляційна матриця (12 змінних)')
plt.tight_layout()
plt.show(block=False)

# Завдання 3: Порівняльний аналіз країн
print("\nВиконується Завдання 3: Лінійні графіки хвиль пандемії...")
countries_5 = ['United States', 'Brazil', 'Germany', 'Japan', 'South Africa']
df_5 = df[df['location'].isin(countries_5)].copy()

# Графік 1:
plt.figure(figsize=(14, 6))
for country in countries_5:
    c_data = df_5[df_5['location'] == country]
    plt.plot(c_data['date'], c_data['new_cases_smoothed'], label=country, linewidth=2)
plt.title('Динаміка нових випадків (new_cases_smoothed) для 5 континентів')
plt.xlabel('Дата')
plt.ylabel('Кількість випадків')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show(block=False)

# Графік 2:
plt.figure(figsize=(14, 6))
for country in countries_5:
    c_data = df_5[df_5['location'] == country]
    plt.plot(c_data['date'], c_data['new_deaths_smoothed'], label=country, linewidth=2)
plt.title('Динаміка нових смертей (new_deaths_smoothed) для 5 континентів')
plt.xlabel('Дата')
plt.ylabel('Кількість смертей')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show(block=False)


# Завдання 4: Інтерактивна візуалізація
print("\nВиконується Завдання 4: Генерація інтерактивного графіка Plotly...")
fig = px.line(df_5,
              x='date',
              y='people_fully_vaccinated_per_hundred',
              color='location',
              title='Динаміка вакцинації (% повністю вакцинованого населення)',
              labels={'people_fully_vaccinated_per_hundred': '% Повністю вакцинованих', 'date': 'Дата', 'location': 'Країна'},
              template='plotly_white')

fig.show()
plt.show()