import csv
import random

# Set random seed for reproducibility
random.seed(42)

# Sri Lankan districts
districts = [
    'Colombo', 'Gampaha', 'Kalutara', 'Kandy', 'Matale', 'Nuwara Eliya',
    'Galle', 'Matara', 'Hambantota', 'Jaffna', 'Kilinochchi', 'Mannar',
    'Vavuniya', 'Mullaitivu', 'Batticaloa', 'Ampara', 'Trincomalee',
    'Kurunegala', 'Puttalam', 'Anuradhapura', 'Polonnaruwa', 'Badulla',
    'Monaragala', 'Ratnapura', 'Kegalle'
]

# Medical conditions (multi-select)
medical_conditions = [
    'None', 'Diabetes', 'Hypertension', 'High Cholesterol', 'Anemia',
    'Asthma', 'PCOS', 'Thyroid Disorder', 'Arthritis', 'IBS',
    'Heart Disease', 'Kidney Disease', 'Osteoporosis'
]

# Allergies/intolerances (multi-select)
allergies = [
    'None', 'Peanuts', 'Tree Nuts', 'Shellfish', 'Fish', 'Eggs',
    'Dairy/Lactose', 'Soy', 'Wheat/Gluten', 'Sesame', 'Sulfites',
    'MSG', 'Nightshades', 'Legumes'
]

# Restrictions
restrictions = [
    'None', 'Vegetarian', 'Vegan', 'Low-sodium <2300mg/day',
    'Low-carb <100g/day', 'Low-fat', 'Gluten-free', 'Dairy-free',
    'Pescatarian', 'Halal', 'Low-purine', 'Low-FODMAP'
]

# Cultural/seasonal preferences
cultural_prefs = [
    'Sri Lankan traditional', 'Rice and curry daily', 'Gotu kola preferred',
    'Jackfruit curry lover', 'Seafood focused', 'Vegetable heavy',
    'Coconut based dishes', 'Less oil preferred', 'Spicy food lover',
    'Traditional breakfast only', 'Hoppers and string hoppers',
    'Lentils and pulses', 'Murunga dishes', 'Bitter gourd preferred',
    'Pumpkin curry lover', 'Mango in season', 'Durian when available'
]

# Cuisine preferences
cuisines = ['Local', 'Mediterranean', 'Mixed', 'Asian fusion']

# Generate age with realistic distribution
def generate_age():
    # More people in working age groups
    age_group = random.choices(
        ['18-25', '26-35', '36-45', '46-55', '56-65'],
        weights=[15, 30, 25, 20, 10]
    )[0]
    
    if age_group == '18-25':
        return random.randint(18, 25)
    elif age_group == '26-35':
        return random.randint(26, 35)
    elif age_group == '36-45':
        return random.randint(36, 45)
    elif age_group == '46-55':
        return random.randint(46, 55)
    else:
        return random.randint(56, 65)

# Generate height based on gender and age
def generate_height(gender, age):
    if gender == 'Male':
        base_height = random.gauss(167, 6)
    elif gender == 'Female':
        base_height = random.gauss(155, 5)
    else:
        base_height = random.gauss(161, 6)
    
    # Slight height decrease with age
    if age > 50:
        base_height -= random.uniform(0, 2)
    
    return round(base_height, 1)

# Generate weight based on height, gender, and age
def generate_weight(height, gender, age):
    # Calculate BMI and adjust
    if gender == 'Male':
        healthy_bmi = random.gauss(23, 2)
    elif gender == 'Female':
        healthy_bmi = random.gauss(22, 2)
    else:
        healthy_bmi = random.gauss(22.5, 2)
    
    # Slight weight increase with age
    if age > 40:
        healthy_bmi += random.uniform(0, 2)
    
    height_m = height / 100
    weight = healthy_bmi * (height_m ** 2)
    
    # Add some variation
    weight *= random.uniform(0.9, 1.1)
    
    return round(weight, 1)

# Generate PAL based on age and weight
def generate_pal(age, weight, height):
    bmi = weight / ((height/100) ** 2)
    
    if bmi > 30 or age > 55:
        # More likely sedentary or lightly active
        return random.choices(
            [1.2, 1.375, 1.55],
            weights=[40, 40, 20]
        )[0]
    elif age < 35:
        # More likely active
        return random.choices(
            [1.375, 1.55, 1.725, 1.9],
            weights=[20, 30, 30, 20]
        )[0]
    else:
        return random.choices(
            [1.2, 1.375, 1.55, 1.725],
            weights=[25, 30, 30, 15]
        )[0]

# Generate medical conditions (multi-select)
def generate_medical_conditions(age):
    conditions = []
    
    # Base chance of having any condition
    if age < 30:
        chance = 0.1
    elif age < 45:
        chance = 0.3
    else:
        chance = 0.6
    
    if random.random() < chance:
        # Determine number of conditions
        num_conditions = random.choices([1, 2, 3], weights=[70, 25, 5])[0]
        available_conditions = [c for c in medical_conditions if c != 'None']
        
        # Age-specific condition probabilities
        if age < 40:
            weights = [5, 3, 2, 8, 10, 7, 4, 2, 6, 1, 1, 3]  # 12 weights for 12 conditions
        else:
            weights = [20, 15, 5, 3, 5, 4, 8, 10, 3, 8, 5, 3]  # 12 weights for 12 conditions
        
        selected = random.choices(available_conditions, weights=weights, k=num_conditions)
        conditions = list(set(selected))  # Remove duplicates
    else:
        conditions = ['None']
    
    return ', '.join(conditions)

# Generate allergies (multi-select)
def generate_allergies():
    chance = 0.35  # 35% chance of having allergies
    if random.random() < chance:
        num_allergies = random.choices([1, 2, 3], weights=[80, 18, 2])[0]
        available_allergies = [a for a in allergies if a != 'None']
        
        # Common allergies in Sri Lanka - weights for 13 allergy types
        weights = [15, 10, 8, 12, 5, 20, 15, 3, 2, 5, 2, 2, 1]
        
        selected = random.choices(available_allergies, weights=weights, k=num_allergies)
        return ', '.join(list(set(selected)))
    else:
        return 'None'

# Generate dietary goal weight
def generate_goal_weight(current_weight, age, gender):
    # Most people want to lose weight
    if random.random() < 0.7:
        # Weight loss goal (realistic 5-15% reduction)
        reduction = random.uniform(0.05, 0.15)
        goal = current_weight * (1 - reduction)
    elif random.random() < 0.5:
        # Muscle gain (slight increase)
        gain = random.uniform(0.02, 0.08)
        goal = current_weight * (1 + gain)
    else:
        # Maintenance (within 2%)
        variation = random.uniform(-0.02, 0.02)
        goal = current_weight * (1 + variation)
    
    return round(goal, 1)

# Generate restrictions
def generate_restrictions(medical_conditions_str):
    conditions = medical_conditions_str.split(', ')
    
    restrictions_list = []
    
    # Medical condition based restrictions
    if 'Diabetes' in conditions:
        restrictions_list.append('Low-carb <100g/day')
    elif 'Hypertension' in conditions or 'Heart Disease' in conditions:
        restrictions_list.append('Low-sodium <2300mg/day')
    elif 'Kidney Disease' in conditions:
        restrictions_list.extend(['Low-sodium <2300mg/day'])
    
    # Random restrictions based on probability
    if not restrictions_list and random.random() < 0.15:
        restrictions_list.append(random.choice([r for r in restrictions if r != 'None']))
    
    # Add vegetarian/vegan based on probability
    if random.random() < 0.12:
        restrictions_list.append(random.choice(['Vegetarian', 'Vegan']))
    
    # If no restrictions, set to None
    if not restrictions_list:
        restrictions_list = ['None']
    
    return ', '.join(list(set(restrictions_list)))

# Generate cultural preferences
def generate_cultural_prefs(district):
    num_prefs = random.choices([1, 2, 3], weights=[40, 40, 20])[0]
    
    # District-specific preferences
    if district in ['Colombo', 'Gampaha', 'Kalutara']:
        # Western province - more diverse
        base_prefs = cultural_prefs + ['International fusion', 'Western options sometimes']
    elif district in ['Jaffna', 'Kilinochchi', 'Mannar']:
        # Northern province
        base_prefs = cultural_prefs + ['Jaffna cuisine specific', 'More vegetarian']
    elif district in ['Hambantota', 'Matara', 'Galle']:
        # Southern province
        base_prefs = cultural_prefs + ['Seafood daily', 'Coconut emphasis']
    else:
        base_prefs = cultural_prefs
    
    selected = random.sample(base_prefs, min(num_prefs, len(base_prefs)))
    return ', '.join(selected)

# Generate cuisine preference
def generate_cuisine(age, district):
    if district in ['Colombo', 'Gampaha'] or age < 35:
        return random.choices(cuisines, weights=[70, 15, 10, 5])[0]
    else:
        return random.choices(cuisines, weights=[85, 5, 8, 2])[0]

# Generate the dataset
def generate_dataset(num_entries=1000):
    data = []
    
    for i in range(num_entries):
        # Basic demographics
        district = random.choice(districts)
        age = generate_age()
        gender = random.choices(['Male', 'Female', 'Other'], weights=[48, 48, 4])[0]
        
        # Physical attributes
        height = generate_height(gender, age)
        weight = generate_weight(height, gender, age)
        
        # Activity and health
        pal = generate_pal(age, weight, height)
        medical = generate_medical_conditions(age)
        allergy = generate_allergies()
        
        # Dietary preferences
        goal_weight = generate_goal_weight(weight, age, gender)
        restriction = generate_restrictions(medical)
        cultural = generate_cultural_prefs(district)
        cuisine = generate_cuisine(age, district)
        
        data.append([
            district,
            age,
            gender,
            round(height, 1),
            round(weight, 1),
            pal,
            medical,
            allergy,
            goal_weight,
            restriction,
            cultural,
            cuisine
        ])
    
    return data

# Generate and save the dataset
dataset = generate_dataset(1000)

# Save to CSV
filename = 'sri_lankan_nutrition_dataset_1000.csv'
with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    
    # Write header
    writer.writerow([
        'District',
        'Age',
        'Gender',
        'Height(cm)',
        'Weight(kg)',
        'Physical_Activity_Level',
        'Medical_Conditions',
        'Allergies_Intolerances',
        'Dietary_Goals(kg)',
        'Restrictions',
        'Cultural_Seasonal_Preferences',
        'Cuisine'
    ])
    
    # Write data
    writer.writerows(dataset)

print(f"✅ Dataset with {len(dataset)} entries saved to {filename}")
print("\n📊 Sample of 5 entries:")
print("=" * 120)
for i in range(5):
    print(f"Entry {i+1}: {dataset[i]}")
print("=" * 120)

print("\n📈 Dataset Statistics:")
print(f"Districts represented: {len(set([row[0] for row in dataset]))}")
print(f"Age range: {min([row[1] for row in dataset])} - {max([row[1] for row in dataset])} years")
print(f"Gender distribution: Male={sum(1 for row in dataset if row[2]=='Male')}, "
      f"Female={sum(1 for row in dataset if row[2]=='Female')}, "
      f"Other={sum(1 for row in dataset if row[2]=='Other')}")
print(f"Average height: {sum([row[3] for row in dataset])/len(dataset):.1f} cm")
print(f"Average weight: {sum([row[4] for row in dataset])/len(dataset):.1f} kg")