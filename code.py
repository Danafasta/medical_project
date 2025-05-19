import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy import stats
from datetime import datetime
from faker import Faker
from jinja2 import Environment, FileSystemLoader
import shutil

# Генерация случайных данных
fake = Faker('ru_RU')

def generate_patient():
    gender = np.random.choice(['М', 'Ж'])
    
    if gender == 'М':
        last_name = fake.last_name_male()
        first_name = fake.first_name_male()
        middle_name = fake.middle_name_male()
    else:
        last_name = fake.last_name_female()
        first_name = fake.first_name_female()
        middle_name = fake.middle_name_female()
    
    full_name = f"{last_name} {first_name} {middle_name}"
    
    return {
        'ФИО': full_name,
        'Пол': gender
    }

# Уникальный данные на основе текущего времени
current_time = datetime.now()
np.random.seed(current_time.microsecond)

# Создаем папку для сохранения результатов
output_dir = f"medical_data_{current_time.strftime('%Y%m%d_%H%M%S')}"
os.makedirs(output_dir, exist_ok=True)

# Настройка стиля графиков
sns.set_theme(style="whitegrid")
sns.set_palette("husl")

# Генерация данных количества пациентов
n_patients = 1000
patients = [generate_patient() for _ in range(n_patients)]
df = pd.DataFrame(patients)

# Медицинские параметры
params = {
    'age': (np.random.randint(35, 55), np.random.randint(10, 20)),
    'weight': (np.random.randint(60, 80), np.random.randint(10, 20)),
    'height': (np.random.randint(160, 180), np.random.randint(8, 12)),
    'systolic_bp': (np.random.randint(110, 130), np.random.randint(10, 20)),
    'diastolic_bp': (np.random.randint(70, 90), np.random.randint(8, 15)),
    'glucose': (np.random.randint(80, 100), np.random.randint(15, 25))
}

# Добавление данных в DataFrame
df['Возраст'] = np.abs(np.random.normal(*params['age'], n_patients)).astype(int)
df['Вес'] = np.abs(np.random.normal(*params['weight'], n_patients)).astype(int)
df['Рост'] = np.abs(np.random.normal(*params['height'], n_patients)).astype(int)
df['Систолическое_давление'] = np.abs(np.random.normal(*params['systolic_bp'], n_patients)).astype(int)
df['Диастолическое_давление'] = np.abs(np.random.normal(*params['diastolic_bp'], n_patients)).astype(int)
df['Уровень_глюкозы'] = np.abs(np.random.normal(*params['glucose'], n_patients)).astype(int)

# Категориальные данные
df['Курение'] = df['Пол'].apply(lambda x: np.random.choice(['Да', 'Нет'], p=[0.3, 0.7]) if x == 'М' else np.random.choice(['Да', 'Нет'], p=[0.15, 0.85]))
df['Диабет'] = df['Пол'].apply(lambda x: np.random.choice(['Да', 'Нет'], p=[0.1, 0.9]) if x == 'М' else np.random.choice(['Да', 'Нет'], p=[0.2, 0.8]))
df['Уровень_холестерина'] = np.random.choice(['Нормальный', 'Повышенный', 'Высокий'], n_patients, p=[0.6, 0.3, 0.1])

# Расчет ИМТ
df['ИМТ'] = df['Вес'] / ((df['Рост'] / 100) ** 2)
df['Категория_ИМТ'] = pd.cut(df['ИМТ'], bins=[0, 18.5, 25, 30, 100], labels=['Недостаточный вес', 'Нормальный вес', 'Избыточный вес', 'Ожирение'])

# Сохранение данных
df.to_csv(f"{output_dir}/medical_data.csv", index=False, encoding='utf-8-sig', sep=';', decimal=',')

# Функции анализа
def analyze_demographics(df):
    gender_dist = df['Пол'].value_counts(normalize=True) * 100
    age_dist = pd.cut(df['Возраст'], bins=[0, 30, 45, 60, 100], labels=['18-30', '31-45', '46-60', '60+']).value_counts(normalize=True) * 100
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.barplot(x=gender_dist.index, y=gender_dist.values)
    plt.title('Распределение по полу')
    plt.ylabel('Процент')
    
    plt.subplot(1, 2, 2)
    sns.barplot(x=age_dist.index, y=age_dist.values, order=['18-30', '31-45', '46-60', '60+'])
    plt.title('Распределение по возрасту')
    plt.ylabel('Процент')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/demographics.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return {'gender_dist': gender_dist.to_dict(), 'age_dist': age_dist.to_dict()}

def analyze_smoking_pressure(df):
    analysis = df.groupby('Курение')[['Систолическое_давление', 'Диастолическое_давление']].mean()
    t_test_sys = stats.ttest_ind(df[df['Курение'] == 'Да']['Систолическое_давление'], df[df['Курение'] == 'Нет']['Систолическое_давление'])
    
    report = []
    conclusions = []
    
    if t_test_sys.pvalue < 0.05:
        diff = analysis.loc['Да', 'Систолическое_давление'] - analysis.loc['Нет', 'Систолическое_давление']
        conclusions.append(f"У курящих людей систолическое давление в среднем на {abs(diff):.1f} мм рт.ст. {'выше' if diff > 0 else 'ниже'} чем у некурящих")
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Курение', y='Систолическое_давление')
    plt.title('Сравнение давления у курящих и некурящих')
    plt.savefig(f"{output_dir}/smoking_pressure.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'detailed_conclusions': conclusions,
        'stats': {
            'mean_sys_smokers': analysis.loc['Да', 'Систолическое_давление'],
            'mean_sys_non_smokers': analysis.loc['Нет', 'Систолическое_давление'],
            'count_smokers': (df['Курение'] == 'Да').sum(),
            'count_non_smokers': (df['Курение'] == 'Нет').sum()
        }
    }

def analyze_age_obesity(df):
    df['Возрастная_группа'] = pd.cut(df['Возраст'], bins=[0, 30, 45, 60, 100], labels=['18-30', '31-45', '46-60', '60+'])
    age_obesity = df.groupby("Возрастная_группа", observed=False)['Категория_ИМТ'].value_counts(normalize=True).unstack()
    
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x='Возрастная_группа', hue='Категория_ИМТ',
                 order=['18-30', '31-45', '46-60', '60+'],
                 hue_order=['Недостаточный вес', 'Нормальный вес', 'Избыточный вес', 'Ожирение'])
    plt.title('Распределение ИМТ по возрастным группам')
    plt.savefig(f"{output_dir}/age_obesity.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    max_group = age_obesity['Ожирение'].idxmax()
    max_percent = age_obesity['Ожирение'].max() * 100
    return f"Наибольший процент ожирения в группе {max_group} ({max_percent:.1f}%)"

def analyze_gender_obesity(df):
    gender_obesity = df.groupby('Пол')['Категория_ИМТ'].value_counts(normalize=True).unstack()
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(gender_obesity, annot=True, fmt='.1%', cmap='YlOrRd')
    plt.title('Распределение ИМТ по полу')
    plt.savefig(f"{output_dir}/gender_obesity.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    male = gender_obesity.loc['М', 'Ожирение'] * 100
    female = gender_obesity.loc['Ж', 'Ожирение'] * 100
    return {'male': male, 'female': female, 'diff': abs(male - female)}

def perform_regression(df, output_dir):
    try:
        X = df[['Возраст', 'ИМТ']]
        y = df['Систолическое_давление']
        model = LinearRegression().fit(X, y)
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.regplot(x='Возраст', y='Систолическое_давление', data=df)
        plt.title('Зависимость давления от возраста')
        
        plt.subplot(1, 2, 2)
        sns.regplot(x='ИМТ', y='Систолическое_давление', data=df)
        plt.title('Зависимость давления от ИМТ')
        
        plt.tight_layout()
        regression_plot_path = os.path.join(output_dir, 'regression_plot.png')
        plt.savefig(regression_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            'age_coef': model.coef_[0],
            'bmi_coef': model.coef_[1],
            'intercept': model.intercept_,
            'r2': model.score(X, y)
        }
    except Exception as e:
        print(f"Ошибка при создании регрессионного графика: {str(e)}")
        return None

def generate_html_report(df, output_dir):
    # Анализ данных
    demographics_result = analyze_demographics(df)
    smoking_result = analyze_smoking_pressure(df)
    age_result = analyze_age_obesity(df)
    gender_result = analyze_gender_obesity(df)
    reg_results = perform_regression(df, output_dir)
    
    # Дополнительные расчеты
    high_pressure = (df['Систолическое_давление'] >= 140).mean() * 100
    obesity_percent = (df['Категория_ИМТ'] == 'Ожирение').mean() * 100
    
    # Расчеты для курящих/некурящих
    smokers_high_pressure = (df[df['Курение'] == 'Да']['Систолическое_давление'] >= 140).mean() * 100
    non_smokers_high_pressure = (df[df['Курение'] == 'Нет']['Систолическое_давление'] >= 140).mean() * 100
    pressure_diff = smoking_result['stats']['mean_sys_smokers'] - smoking_result['stats']['mean_sys_non_smokers']
    relative_risk = smokers_high_pressure / non_smokers_high_pressure if non_smokers_high_pressure != 0 else 0
    
    # Расчеты по полу
    male_stats = df[df['Пол'] == 'М']
    female_stats = df[df['Пол'] == 'Ж']
    male_avg_bmi = male_stats['ИМТ'].mean()
    female_avg_bmi = female_stats['ИМТ'].mean()
    male_overweight = (male_stats['Категория_ИМТ'].isin(['Избыточный вес', 'Ожирение'])).mean() * 100
    female_overweight = (female_stats['Категория_ИМТ'].isin(['Избыточный вес', 'Ожирение'])).mean() * 100
    
    # Возрастные группы ИМТ
    age_bmi_corr = df['Возраст'].corr(df['ИМТ'])
    
    # Копируем CSS и JS файлы
    try:
        shutil.copy('styles.css', output_dir)
        shutil.copy('script.js', output_dir)
    except Exception as e:
        print(f"Ошибка при копировании файлов: {e}")
    
    # Генерация HTML
    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template('medical_report_template.html')
    
    context = {
        'report_date': current_time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_patients': f"{len(df):,}",
        'avg_age': f"{df['Возраст'].mean():.1f} ± {df['Возраст'].std():.1f} лет",
        'gender_ratio': f"{(df['Пол'] == 'М').sum():,} / {(df['Пол'] == 'Ж').sum():,}",
        'gender_dist': {
            'male': f"{demographics_result['gender_dist'].get('М', 0):.1f}",
            'female': f"{demographics_result['gender_dist'].get('Ж', 0):.1f}"
        },
        'age_dist': {
            'age_18_30': f"{demographics_result['age_dist'].get('18-30', 0):.1f}",
            'age_31_45': f"{demographics_result['age_dist'].get('31-45', 0):.1f}",
            'age_46_60': f"{demographics_result['age_dist'].get('46-60', 0):.1f}",
            'age_60_plus': f"{demographics_result['age_dist'].get('60+', 0):.1f}"
        },
        'smoking_percent': f"{(df['Курение'] == 'Да').mean() * 100:.1f}",
        'diabetes_percent': f"{(df['Диабет'] == 'Да').mean() * 100:.1f}",
        'avg_bmi': f"{df['ИМТ'].mean():.1f}",
        'avg_pressure': f"{df['Систолическое_давление'].mean():.1f}/{df['Диастолическое_давление'].mean():.1f} мм рт.ст.",
        'smoking_conclusions': smoking_result['detailed_conclusions'],
        'age_obesity_result': age_result,
        'male_obesity': f"{gender_result['male']:.1f}",
        'female_obesity': f"{gender_result['female']:.1f}",
        'obesity_diff': f"{gender_result['diff']:.1f}",
        
        # Основные показатели
        'high_pressure_percent': f"{high_pressure:.1f}",
        'obesity_percent': f"{obesity_percent:.1f}",
        
        # Статистика по курению
        'smokers_count': smoking_result['stats']['count_smokers'],
        'smokers_pressure': f"{smoking_result['stats']['mean_sys_smokers']:.1f}/{df[df['Курение'] == 'Да']['Диастолическое_давление'].mean():.1f}",
        'smokers_high_pressure': f"{smokers_high_pressure:.1f}",
        'non_smokers_count': smoking_result['stats']['count_non_smokers'],
        'non_smokers_pressure': f"{smoking_result['stats']['mean_sys_non_smokers']:.1f}/{df[df['Курение'] == 'Нет']['Диастолическое_давление'].mean():.1f}",
        'non_smokers_high_pressure': f"{non_smokers_high_pressure:.1f}",
        'pressure_diff': f"{abs(pressure_diff):.1f}",
        'relative_risk': f"{relative_risk:.1f}",
        
        # Статистика по полу
        'male_avg_bmi': f"{male_avg_bmi:.1f}",
        'male_overweight': f"{male_overweight:.1f}",
        'female_avg_bmi': f"{female_avg_bmi:.1f}",
        'female_overweight': f"{female_overweight:.1f}",
        'age_bmi_correlation': f"{age_bmi_corr:.2f}",
        
        # Дополнительные данные
        'max_bmi_age_group': age_result.split('(')[0].replace('Наибольший процент ожирения в группе ', ''),
        'min_bmi_age_group': '18-30'
    }
    
    if reg_results:
        context.update({
            'regression_success': True,
            'age_coef': f"{reg_results['age_coef']:.2f}",
            'bmi_coef': f"{reg_results['bmi_coef']:.2f}",
            'intercept': f"{reg_results['intercept']:.2f}",
            'r_squared': f"{reg_results['r2']:.3f}"
        })
    else:
        context['regression_success'] = False
    
    with open(f"{output_dir}/medical_report.html", "w", encoding="utf-8") as f:
        f.write(template.render(context))

# Запуск генерации отчета
generate_html_report(df, output_dir)

print(f"\nОтчет успешно сгенерирован в папке: {output_dir}")
print("Пример данных:")
print(df[['ФИО', 'Возраст', 'Пол', 'Категория_ИМТ']].sample(3))