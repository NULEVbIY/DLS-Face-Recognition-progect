import pandas as pd
import random
import os
from collections import Counter

# Загружаем данные
base_path = r"C:\Users\Public\Documents\Face recognition"
identity_path = os.path.join(base_path, "identity_CelebA.txt")
attr_path = os.path.join(base_path, "list_attr_celeba.csv")

identity_df = pd.read_csv(identity_path, sep=' ', header=None, names=['image_id', 'person_id'])
all_attr = pd.read_csv(attr_path)

# Объединяем identity с атрибутами
merged = pd.merge(identity_df, all_attr, on='image_id')

# Разделяем на мужчин и женщин
male_data = merged[merged['Male'] == 1]
female_data = merged[merged['Male'] == -1]

# для фильтрации по атрибутам
def filter_by_attributes(data, attributes_dict):
    
    filtered = data.copy()
    for attr, value in attributes_dict.items():
        if attr in filtered.columns:
            filtered = filtered[filtered[attr] == value]
    return filtered

# Создаем подгруппы
def create_subgroups(data, subgroup_name):
   
    subgroups = {}
    
    # Комбинации атрибутов
    age_groups = [('Young', 1), ('Young', -1)]  # молодые / не молодые
    
    # Цвет волос 
    hair_colors = [
        ('Black_Hair', 1),    # темные волосы
        ('Blond_Hair', 1),    # блондинистые волосы  
        ('Brown_Hair', 1),    # русые волосы
        ('Gray_Hair', 1),     # седые волосы
    ]
    
    smile_groups = [('Smiling', 1), ('Smiling', -1)]  # улыбающиеся / не улыбающиеся
    
    # Создаем подгруппы
    subgroup_id = 0
    for age_attr, age_val in age_groups:
        for hair_attr, hair_val in hair_colors:
            for smile_attr, smile_val in smile_groups:
                filtered = data[
                    (data[age_attr] == age_val) &
                    (data[hair_attr] == hair_val) &
                    (data[smile_attr] == smile_val)
                ]
                
                if len(filtered) > 0:
                    subgroups[f"{subgroup_name}_sub{subgroup_id:02d}"] = {
                        'data': filtered,
                        'filters': {
                            'age': (age_attr, age_val),
                            'hair': (hair_attr, hair_val),
                            'smile': (smile_attr, smile_val)
                        },
                        'person_count': filtered['person_id'].nunique(),
                        'photo_count': len(filtered)
                    }
                    subgroup_id += 1
    
    return subgroups

# Создаем подгруппы для мужчин и женщин
male_subgroups = create_subgroups(male_data, 'male')

female_subgroups = create_subgroups(female_data, 'female')

# Находим людей с ≥10 фото в каждой подгруппе
def find_people_with_min_photos(subgroups, min_photos=10):
    """
    Находит людей с min_photos фото в каждой подгруппе
    """
    result = {}
    
    for subgroup_name, info in subgroups.items():
        data = info['data']
        
        # Считаем фото для каждого человека
        person_counts = data.groupby('person_id').size()
        
        # Люди с ≥min_photos фото
        eligible_people = person_counts[person_counts >= min_photos].index.tolist()
        
        if eligible_people:
            result[subgroup_name] = {
                'eligible_people': eligible_people,
                'count': len(eligible_people),
                'avg_photos': person_counts[person_counts >= min_photos].mean(),
                'filters': info['filters']}
    return result

male_eligible = find_people_with_min_photos(male_subgroups, 10)
female_eligible = find_people_with_min_photos(female_subgroups, 10)

# Отбираем людей для датасета
def select_people_for_dataset(eligible_groups, n_people_needed):
    selected_people = []
    selected_info = []
    
    # Собираем всех подходящих людей
    all_candidates = []
    for subgroup_name, info in eligible_groups.items():
        for person_id in info['eligible_people']:
            all_candidates.append({
                'person_id': person_id,
                'subgroup': subgroup_name,
                'filters': info['filters']
            })
    
    # Убираем дубликаты
    unique_candidates = {}
    for candidate in all_candidates:
        person_id = candidate['person_id']
        if person_id not in unique_candidates:
            unique_candidates[person_id] = candidate
    
    
    # Если кандидатов недостаточно
    if len(unique_candidates) < n_people_needed:
        # Берем всех кого есть
        selected_people = list(unique_candidates.keys())[:n_people_needed]
    else:
        # Выбираем случайно
        selected_ids = random.sample(list(unique_candidates.keys()), n_people_needed)
        selected_people = selected_ids
    
    return selected_people

# Создаем датасет 15000 фото (1000 человек по 15 фото)
def create_final_dataset(selected_people, total_photos=15000, photos_per_person=15):
    all_selected_photos = []
    
    for person_id in selected_people:
        # Все фото этого человека
        person_photos = merged[merged['person_id'] == person_id]['image_id'].tolist()
        
        if len(person_photos) >= photos_per_person:
            selected = random.sample(person_photos, photos_per_person)
        else:
            selected = person_photos
        
        all_selected_photos.extend(selected)
    
    # Обрезаем если получилось больше
    all_selected_photos = all_selected_photos[:total_photos]
    
    return all_selected_photos

# Нужно 1000 человек (500 мужчин, 500 женщин)
n_people_total = 1000
n_male_needed = 500
n_female_needed = 500

# Отбираем мужчин
selected_male = select_people_for_dataset(male_eligible, n_male_needed)

# Отбираем женщин  
selected_female = select_people_for_dataset(female_eligible, n_female_needed)

# Объединяем
selected_people = selected_male + selected_female

# Создаем датасет
final_dataset = create_final_dataset(selected_people, 15000, 15)

# Проверяем баланс финального датасета
def check_final_balance(dataset_photos, merged_data):
    dataset_df = merged_data[merged_data['image_id'].isin(dataset_photos)]
    
    # Основные атрибуты
    attributes_to_check = ['Male', 'Young', 'Smiling', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']
    
    for attr in attributes_to_check:
        if attr in dataset_df.columns:
            pos = (dataset_df[attr] == 1).sum()
            total = len(dataset_df)
            pct = pos / total * 100 if total > 0 else 0
            
            # Определяем название атрибута для вывода
            attr_names = {
                'Male': 'Мужчины',
                'Young': 'Молодые',
                'Smiling': 'Улыбающиеся',
                'Black_Hair': 'Темные волосы',
                'Blond_Hair': 'Блондинистые волосы',
                'Brown_Hair': 'Русые волосы',
                'Gray_Hair': 'Седые волосы'
            }
            
            attr_name = attr_names.get(attr, attr)
            print(f"{attr_name:20} {pos:5d} ({pct:5.1f}%)")
    
    # Проверяем уникальных людей
    unique_people = dataset_df['person_id'].nunique()
    print(f"\nУникальных людей в датасете: {unique_people}")
    print(f"Среднее фото на человека: {len(dataset_photos)/unique_people:.1f}")

# Проверяем
check_final_balance(final_dataset, merged)

# Сохраняем результат
def save_dataset(dataset_photos, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        for photo in dataset_photos:
            f.write(f"{photo}\n")
    

# Сохраняем
save_dataset(final_dataset, os.path.join(base_path, "my_dataset_15000.txt"))
