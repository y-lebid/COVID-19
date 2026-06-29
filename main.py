import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Завдання 2: Завантаження та первинний огляд

df = pd.read_csv('covid_data.csv')

print("Перші 5 рядків:")
print(df.head())

print("\nІнформація про датасет:")
df.info()

print("\nНазви стовпців:")
print(df.columns)


# Завдання 3: Очищення даних

print("\nКількість відсутніх значень:")
print(df.isnull().sum())

df = df.drop_duplicates()

df['date'] = pd.to_datetime(df['date'])


# Завдання 4: Статистичний огляд

key_metrics = ['total_cases', 'new_cases', 'total_deaths', 'new_deaths']
print("\nСтатистика ключових показників:")
print(df[key_metrics].describe())


# Завдання 5: Візуалізація

df_ukraine = df[df['location'] == 'Ukraine'].sort_values('date')

plt.figure(figsize=(12, 6))
plt.plot(df_ukraine['date'], df_ukraine['total_cases'], label='Загальна к-ть випадків', color='blue')
plt.plot(df_ukraine['date'], df_ukraine['total_deaths'], label='Загальна к-ть смертей', color='red')
plt.title('Динаміка випадків COVID-19 та смертей в Україні')
plt.xlabel('Дата')
plt.ylabel('Кількість')
plt.legend()
plt.grid(True)
plt.show(block=False)

df_latest_correct = df.groupby(['continent', 'location'])['total_cases'].max().reset_index()
continent_cases = df_latest_correct.groupby('continent')['total_cases'].sum().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=continent_cases.index, y=continent_cases.values, hue=continent_cases.index, palette='viridis', legend=False)
plt.title('Загальна кількість випадків COVID-19 по континентах')
plt.xlabel('Континент')
plt.ylabel('Загальна кількість випадків (сотні мільйонів)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show(block=False)


# Завдання 6: Очищення та обробка даних

print("\n--- Завдання 6 ---")
print("Пропуски до обробки:\n", df[['new_cases', 'new_deaths', 'total_cases']].isnull().sum())
df = df.sort_values(by=['location', 'date'])

df['new_cases'] = df['new_cases'].fillna(0)
df['new_deaths'] = df['new_deaths'].fillna(0)
df['total_cases'] = df.groupby('location')['total_cases'].transform(lambda x: x.ffill().fillna(0))

duplicates_count = df.duplicated().sum()
print(f"Знайдено та видалено дублікатів: {duplicates_count}")
df = df.drop_duplicates()


# Завдання 7: Трансформація даних

print("\n--- Завдання 7 ---")
df['location_encoded'] = df['iso_code']

df['growth_rate_new_cases'] = df.groupby('location')['new_cases'].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
df['growth_rate_new_deaths'] = df.groupby('location')['new_deaths'].pct_change().replace([np.inf, -np.inf], 0).fillna(0)

df.to_csv('covid_data_cleaned.csv', index=False)
print("Очищені та трансформовані дані збережено у 'covid_data_cleaned.csv'.")


# Завдання 8: Кореляційний аналіз

print("\n--- Завдання 8 ---")
corr_cols = ['new_cases', 'new_deaths', 'total_cases', 'population', 'gdp_per_capita']
corr_matrix = df[corr_cols].corr()

print("Кореляційна матриця:\n", corr_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Кореляційна матриця числових змінних')
plt.tight_layout()
plt.show(block=False)


# Завдання 9: Розподіл даних

print("\n--- Завдання 9 ---")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(df['total_cases'], bins=50, ax=axes[0], color='blue', log_scale=(False, True))
axes[0].set_title('Розподіл загальної кількості випадків (Log Scale)')

sns.histplot(df['total_deaths'].dropna(), bins=50, ax=axes[1], color='red', log_scale=(False, True))
axes[1].set_title('Розподіл загальної кількості смертей (Log Scale)')
plt.tight_layout()
plt.show(block=False)

plt.figure(figsize=(10, 6))
sns.boxplot(x='continent', y='total_deaths_per_million', data=df[df['continent'].notna()])
plt.title('Розподіл смертей на мільйон за континентами')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show(block=False)

pairplot_cols = ['total_cases', 'total_deaths', 'total_vaccinations', 'population']
sns.pairplot(df[pairplot_cols].dropna().sample(n=1000, random_state=42), diag_kind='kde')
plt.suptitle('Pairplot для ключових змінних (вибірка 1000 записів)', y=1.02)
plt.show(block=False)


# Завдання 10: Аналіз трендів

print("\n--- Завдання 10 ---")
target_countries = ['United States', 'Ukraine', 'Germany']
df_targets = df[df['location'].isin(target_countries)]

fig, axes = plt.subplots(2, 1, figsize=(12, 10))

for country in target_countries:
    c_data = df_targets[df_targets['location'] == country]
    axes[0].plot(c_data['date'], c_data['new_cases'], label=country)
    axes[1].plot(c_data['date'], c_data['new_deaths'], label=country, linestyle='--')

axes[0].set_title('Тренд нових випадків (США, Україна, Німеччина)')
axes[0].legend()
axes[0].grid(True)

axes[1].set_title('Тренд нових смертей (США, Україна, Німеччина)')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show(block=False)

df_latest_per_country = df[df['continent'].notna()].groupby('location').last().reset_index()
top_10 = df_latest_per_country.nlargest(10, 'total_cases_per_million')


plt.figure(figsize=(10, 6))
sns.barplot(x='total_cases_per_million', y='location', data=top_10, hue='location', palette='magma', legend=False)
plt.title('Топ-10 країн за кількістю випадків на мільйон (на останню дату)')
plt.xlabel('Випадки на мільйон')
plt.ylabel('Країна')
plt.tight_layout()
plt.show(block=False)

print("\nДати з максимальними та мінімальними значеннями:")
for country in target_countries:
    c_data = df[df['location'] == country].copy()
    if not c_data.empty:
        max_c = c_data.loc[c_data['new_cases'].idxmax()]
        min_c = c_data.loc[c_data['new_cases'].idxmin()]
        max_d = c_data.loc[c_data['new_deaths'].idxmax()]
        min_d = c_data.loc[c_data['new_deaths'].idxmin()]

        print(f"\n{country}:")
        print(f"  Макс. нових випадків: {max_c['new_cases']} (Дата: {max_c['date'].date()})")
        print(f"  Мін. нових випадків:  {min_c['new_cases']} (Дата: {min_c['date'].date()})")
        print(f"  Макс. нових смертей:  {max_d['new_deaths']} (Дата: {max_d['date'].date()})")
        print(f"  Мін. нових смертей:   {min_d['new_deaths']} (Дата: {min_d['date'].date()})")

print("Усі завдання виконано! Графіки на екрані.")
plt.show()