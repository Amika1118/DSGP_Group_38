import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import ast
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# --- 1. INITIALIZE SESSION STATE ---
if 'profiles' not in st.session_state:
    st.session_state.profiles = []
if 'last_result' not in st.session_state:
    st.session_state.last_result = None
if 'saved_households' not in st.session_state:
    st.session_state.saved_households = {}
if 'feedback_history' not in st.session_state:
    st.session_state.feedback_history = []
if 'language' not in st.session_state:
    st.session_state.language = 'English'
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'models' not in st.session_state:
    st.session_state.models = {}

# --- 2. CONFIGURATION & CONSTANTS ---
st.set_page_config(
    page_title="🌱 Sri Lanka Veggie Advisor (CotD 2024)",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sri Lanka districts (all 25)
SRI_LANKA_DISTRICTS = [
    'Ampara', 'Anuradhapura', 'Badulla', 'Batticaloa', 'Colombo',
    'Galle', 'Gampaha', 'Hambantota', 'Jaffna', 'Kalutara',
    'Kandy', 'Kegalle', 'Kilinochchi', 'Mannar', 'Matale',
    'Matara', 'Monaragala', 'Mullaitivu', 'Nuwara Eliya',
    'Polonnaruwa', 'Puttalam', 'Ratnapura', 'Trincomalee', 'Vavuniya'
]

# District risk factors for affordability calculation
DISTRICT_RISKS = {
    'Ampara': 0.42, 'Anuradhapura': 0.38, 'Badulla': 0.35,
    'Batticaloa': 0.45, 'Colombo': 0.28, 'Galle': 0.32,
    'Gampaha': 0.30, 'Hambantota': 0.37, 'Jaffna': 0.40,
    'Kalutara': 0.33, 'Kandy': 0.34, 'Kegalle': 0.32,
    'Kilinochchi': 0.44, 'Mannar': 0.43, 'Matale': 0.36,
    'Matara': 0.35, 'Monaragala': 0.39, 'Mullaitivu': 0.45,
    'Nuwara Eliya': 0.31, 'Polonnaruwa': 0.37, 'Puttalam': 0.36,
    'Ratnapura': 0.34, 'Trincomalee': 0.41, 'Vavuniya': 0.39
}

# CotD 2024 Ratios
COTD_RATIOS = {
    "Child (<8 years)": 0.16,
    "Adolescent Girl (14-15)": 0.30,
    "Adult Male (30-59)": 0.29,
    "Adult Female (30-59)": 0.25,
    "Other": 0.25
}

# CotD 2024 Priority Vegetables by District
COTD_VEGGIES = {
    'Ampara': ['Bean (yard long)', 'Kathurumurunga'],
    'Anuradhapura': ['Bean (yard long)', 'Kathurumurunga'],
    'Badulla': ['Bean (yard long)', 'Eggplant', 'Kankun', 'Kathurumurunga'],
    'Batticaloa': ['Bean (yard long)', 'Kathurumurunga'],
    'Colombo': ['Bean (yard long)', 'Kathurumurunga'],
    'Galle': ['Bean (yard long)', 'Eggplant', 'Kathurumurunga', 'Mukunuwenna', 'Okra'],
    'Gampaha': ['Bean (yard long)', 'Eggplant', 'Kankun', 'Kathurumurunga', 'Mukunuwenna', 'Okra'],
    'Hambantota': ['Banana blossom', 'Bean (yard long)', 'Kathurumurunga', 'Okra'],
    'Jaffna': ['Ash Plantain', 'Bean (yard long)', 'Eggplant', 'Kankun', 'Kathurumurunga', 'Okra'],
    'Kalutara': ['Bean (yard long)', 'Kathurumurunga'],
    'Kandy': ['Bean (yard long)', 'Kathurumurunga', 'Okra'],
    'Kegalle': ['Bean (yard long)', 'Kathurumurunga'],
    'Kilinochchi': ['Bean (yard long)', 'Kathurumurunga'],
    'Mannar': ['Bean (yard long)', 'Eggplant', 'Kathurumurunga', 'Okra'],
    'Matale': ['Bean (yard long)', 'Kathurumurunga'],
    'Matara': ['Bean (yard long)', 'Kathurumurunga'],
    'Monaragala': ['Bean (yard long)', 'Kathurumurunga'],
    'Mullaitivu': ['Bean (yard long)', 'Kathurumurunga'],
    'Nuwara Eliya': ['Bean (yard long)', 'Kathurumurunga'],
    'Polonnaruwa': ['Bean (yard long)', 'Kathurumurunga'],
    'Puttalam': ['Bean (yard long)', 'Kathurumurunga'],
    'Ratnapura': ['Bean (yard long)', 'Kathurumurunga'],
    'Trincomalee': ['Bean (yard long)', 'Kathurumurunga'],
    'Vavuniya': ['Bean (yard long)', 'Kathurumurunga']
}

# USDA to CotD mapping
USDA_TO_COTD_MAP = {
    'BEANS, GREEN': 'Bean (yard long)',
    'BEANS, SNAP, RAW': 'Bean (yard long)',
    'MORINGA LEAVES, RAW': 'Kathurumurunga',
    'SWAMP CABBAGE': 'Kankun',
    'WATERSPINACH, RAW': 'Kankun',
    'BEET GREENS, RAW': 'Mukunuwenna',
    'OKRA, RAW': 'Okra',
    'EGGPLANT, RAW': 'Eggplant',
    'PLANTAIN, GREEN, RAW': 'Ash Plantain',
    'BANANA, RAW': 'Banana blossom',
}

# Activity level multipliers
ACTIVITY_MULTIPLIERS = {
    'Sedentary': 1.2,
    'Light (1-3 days/week)': 1.375,
    'Moderate (3-5 days/week)': 1.55,
    'Active (6-7 days/week)': 1.725,
    'Very Active (twice daily)': 1.9
}

# Medical conditions and contraindications
MEDICAL_CONDITIONS = {
    'Diabetes': {'avoid': ['CARROT', 'BEETROOT', 'SWEET POTATO'], 'recommend': ['BITTER GOURD', 'STRING BEANS']},
    'Hypertension': {'avoid': ['HIGH SODIUM'], 'recommend': ['POTASSIUM']},
    'Anemia': {'avoid': [], 'recommend': ['SPINACH', 'BEET GREENS', 'MORINGA']},
    'Pregnancy': {'avoid': ['PAPAYA', 'PINEAPPLE'], 'recommend': ['LEAFY GREENS', 'LEGUMES']},
    'Kidney Disease': {'avoid': ['POTASSIUM', 'SPINACH'], 'recommend': ['CABBAGE', 'CAULIFLOWER']},
    'Heart Disease': {'avoid': ['COCONUT'], 'recommend': ['GARLIC', 'ONION']},
    'Obesity': {'avoid': ['STARCHY'], 'recommend': ['LEAFY GREENS', 'CUCUMBER']},
    'Underweight': {'avoid': [], 'recommend': ['STARCHY', 'LEGUMES']},
}

# Allergy mappings
ALLERGY_MAPPINGS = {
    'Nightshades': ['TOMATO', 'EGGPLANT', 'PEPPER', 'POTATO', 'GOJI BERRY'],
    'Cruciferous': ['CABBAGE', 'BROCCOLI', 'CAULIFLOWER', 'KALE', 'BRUSSELS'],
    'Leafy Greens': ['SPINACH', 'LETTUCE', 'KALE', 'SWISS CHARD'],
    'Legumes': ['BEAN', 'PEA', 'LENTIL', 'CHICKPEA'],
    'Root Vegetables': ['CARROT', 'POTATO', 'BEET', 'RADISH', 'TURNIP'],
}

# --- 3. DATA LOADING FUNCTIONS (NO CACHE FOR MODELS) ---
@st.cache_data
def load_data_files():
    """Load data files without models"""
    veggies = pd.read_csv('vegetables_USDA.csv')
    recipes = pd.read_csv('sri_lankan_recipes_comprehensive.csv')
    veg_season = pd.read_csv('vegetable_seasonality_sri_lanka_comprehensive.csv')
    
    # Core nutrients for ranking
    nutrients = ['Energ_Kcal', 'Protein_(g)', 'Fiber_TD_(g)', 'Iron_(mg)', 
                 'Potassium_(mg)', 'Vit_C_(mg)', 'Vit_A_RAE', 'Calcium_(mg)']
    available_nutrients = [col for col in nutrients if col in veggies.columns]
    
    # Clean nutrient columns
    for col in available_nutrients:
        veggies[col] = pd.to_numeric(veggies[col], errors='coerce').fillna(0)
    
    # Add local names
    veggies['Sinhala_Name'] = veggies['Shrt_Desc'].apply(lambda x: get_sinhala_name(x))
    veggies['Tamil_Name'] = veggies['Shrt_Desc'].apply(lambda x: get_tamil_name(x))
    
    # Clean recipe data
    recipes['main_veg_clean'] = recipes['vegetables_usda'].apply(
        lambda x: [v.strip().upper() for v in ast.literal_eval(x)] if isinstance(x, str) else []
    )
    
    return veggies, recipes, veg_season, available_nutrients

def load_ml_models():
    """Load ML models separately (not cached)"""
    models = {}
    
    # Check for model files
    model_files = {
        'xgboost': ['xgboost_model.json', 'xgboost_model.model', 'model.pkl'],
        'spn': ['spn_model.pkl', 'spn_model.joblib']
    }
    
    # Try to load XGBoost model
    xgb_loaded = False
    for model_file in model_files['xgboost']:
        if os.path.exists(model_file):
            try:
                if model_file.endswith('.json'):
                    models['xgboost'] = xgb.Booster()
                    models['xgboost'].load_model(model_file)
                elif model_file.endswith('.model'):
                    models['xgboost'] = xgb.Booster()
                    models['xgboost'].load_model(model_file)
                elif model_file.endswith('.pkl'):
                    with open(model_file, 'rb') as f:
                        models['xgboost'] = pickle.load(f)
                xgb_loaded = True
                st.session_state.models['xgboost'] = models['xgboost']
                break
            except Exception as e:
                st.warning(f"Failed to load XGBoost model from {model_file}: {e}")
    
    if not xgb_loaded:
        st.info("ℹ️ XGBoost model not found. Using similarity-based recommendations.")
        models['xgboost'] = None
    
    # Load SPN model if available
    spn_loaded = False
    for model_file in model_files['spn']:
        if os.path.exists(model_file):
            try:
                with open(model_file, 'rb') as f:
                    models['spn'] = pickle.load(f)
                spn_loaded = True
                st.session_state.models['spn'] = models['spn']
                break
            except Exception as e:
                st.warning(f"Failed to load SPN model from {model_file}: {e}")
    
    if not spn_loaded:
        models['spn'] = None
    
    st.session_state.models_loaded = True
    st.session_state.models = models
    
    return models

def get_sinhala_name(english_name):
    """Translate English vegetable names to Sinhala"""
    sinhala_dict = {
        'CARROT': 'කැරට්',
        'BEANS': 'බෝංචි',
        'CABBAGE': 'ගෝවා',
        'TOMATO': 'තක්කාලි',
        'ONION': 'ලූනු',
        'POTATO': 'අර්තාපල්',
        'BRINJAL': 'වම්බටු',
        'CUCUMBER': 'පිපිංචා',
        'PUMPKIN': 'වට්ටක්කා',
        'SPINACH': 'නිවිති',
        'BEETROOT': 'බීට්රූට්',
        'RADISH': 'රාබු',
        'OKRA': 'බණ්ඩක්කා',
        'LEEK': 'ලීක්ස්',
        'LETTUCE': 'ලෙටිස්',
        'BITTER GOURD': 'කරවිල',
    }
    for key, value in sinhala_dict.items():
        if key in english_name.upper():
            return value
    return english_name.split(',')[0]

def get_tamil_name(english_name):
    """Translate English vegetable names to Tamil"""
    tamil_dict = {
        'CARROT': 'கேரட்',
        'BEANS': 'பீன்ஸ்',
        'CABBAGE': 'முட்டைகோஸ்',
        'TOMATO': 'தக்காளி',
        'ONION': 'வெங்காயம்',
        'POTATO': 'உருளைக்கிழங்கு',
        'BRINJAL': 'கத்தரிக்காய்',
        'CUCUMBER': 'வெள்ளரிக்காய்',
        'PUMPKIN': 'பூசணிக்காய்',
        'SPINACH': 'பசலைக்கீரை',
        'BEETROOT': 'பீட்ரூட்',
        'RADISH': 'முள்ளங்கி',
        'OKRA': 'வெண்டைக்காய்',
        'LEEK': 'லீக்',
        'LETTUCE': 'லெட்டுஸ்',
        'BITTER GOURD': 'பாகற்காய்',
    }
    for key, value in tamil_dict.items():
        if key in english_name.upper():
            return value
    return english_name.split(',')[0]

# Load data files (cached)
veggies, recipes, veg_season, available_nutrients = load_data_files()

# --- 4. CORE LOGIC FUNCTIONS ---
def calculate_bmr(weight, height, age, gender):
    """Calculate Basal Metabolic Rate using Mifflin-St Jeor"""
    if gender.lower() == 'male':
        bmr = (10 * weight) + (6.25 * height) - (5 * age) + 5
    else:
        bmr = (10 * weight) + (6.25 * height) - (5 * age) - 161
    return bmr

def calculate_tee(bmr, activity_level):
    """Calculate Total Energy Expenditure"""
    return bmr * ACTIVITY_MULTIPLIERS.get(activity_level, 1.55)

def determine_cotd_role(age, gender):
    """Determine CotD consumption ratio role"""
    if age < 8:
        return "Child (<8 years)", 0.16
    elif 14 <= age <= 15 and gender.lower() == 'female':
        return "Adolescent Girl (14-15)", 0.30
    elif 30 <= age <= 59:
        if gender.lower() == 'male':
            return "Adult Male (30-59)", 0.29
        else:
            return "Adult Female (30-59)", 0.25
    else:
        return "Other", 0.25

def clean_string_for_matching(s):
    """Clean string for matching"""
    if pd.isna(s):
        return ""
    s = str(s).replace('"', '').replace("'", "").replace(',', ' ').replace('RAW', '')
    return "".join(s.split()).upper()

def is_in_season(veg_name, district, month=None):
    """Check if vegetable is in season"""
    if month is None:
        month = datetime.now().month
    
    veg_clean = clean_string_for_matching(veg_name)
    season_match = veg_season[
        (veg_season['usda_code'].apply(clean_string_for_matching) == veg_clean) &
        (veg_season['district'] == district)
    ]
    
    if season_match.empty:
        return None  # Unknown
    
    row = season_match.iloc[0]
    is_maha = month in [10, 11, 12, 1, 2, 3]
    
    if is_maha and row['maha_season'] == 'YES':
        return 'Maha Season'
    elif not is_maha and row['yala_season'] == 'YES':
        return 'Yala Season'
    else:
        return 'Off Season'

def scale_ingredients(ingredients_str, total_weight):
    """Scale recipe ingredients for household size"""
    try:
        if pd.isna(ingredients_str):
            return "Ingredients not specified"
            
        items = str(ingredients_str).split(',')
        scaled = []
        
        for item in items:
            item = item.strip()
            if not item:
                continue
                
            # Try to parse quantity
            import re
            match = re.match(r'^(\d*\.?\d+)\s*([a-zA-Z]*)\s*(.*)', item)
            if match:
                quantity = float(match.group(1))
                unit = match.group(2).lower()
                ingredient_name = match.group(3).strip().lower()
                
                # Scale quantity
                scaled_quantity = quantity * total_weight
                
                # Round appropriately
                if unit in ['kg', 'kilogram'] and scaled_quantity < 1:
                    scaled_quantity *= 1000
                    unit = 'g'
                elif unit in ['l', 'liter', 'lt'] and scaled_quantity < 1:
                    scaled_quantity *= 1000
                    unit = 'ml'
                
                scaled_item = f"{scaled_quantity:.0f if scaled_quantity.is_integer() else scaled_quantity:.1f}{unit} {ingredient_name.title()}"
                scaled.append(scaled_item)
            else:
                scaled.append(item.title())
        
        return ', '.join(scaled)
    except Exception as e:
        return f"Error scaling ingredients: {str(e)}"

def aggregate_household(profiles):
    """Aggregate household nutritional needs"""
    if not profiles:
        return None, {}
    
    total_weight = 0
    total_tee = 0
    total_calories = 0
    aggregate_target = np.zeros(len(available_nutrients))
    conditions = set()
    allergies = set()
    household_notes = []
    
    for p in profiles:
        bmr = calculate_bmr(p['weight'], p['height'], p['age'], p['gender'])
        tee = calculate_tee(bmr, p['activity'])
        
        role, ratio = determine_cotd_role(p['age'], p['gender'])
        
        # Calculate nutrient targets
        member_target = np.array([
            tee * 0.2 / 5 if n == 'Energ_Kcal' else 1.0 
            for n in available_nutrients
        ])
        
        # Adjust for medical conditions
        for condition in p.get('conditions', []):
            if condition in MEDICAL_CONDITIONS:
                # Increase need for recommended nutrients
                for rec in MEDICAL_CONDITIONS[condition]['recommend']:
                    if rec == 'POTASSIUM':
                        potassium_idx = available_nutrients.index('Potassium_(mg)') if 'Potassium_(mg)' in available_nutrients else -1
                        if potassium_idx >= 0:
                            member_target[potassium_idx] *= 1.5
                    elif rec == 'LEAFY GREENS':
                        # Boost iron and vitamins
                        iron_idx = available_nutrients.index('Iron_(mg)') if 'Iron_(mg)' in available_nutrients else -1
                        vitc_idx = available_nutrients.index('Vit_C_(mg)') if 'Vit_C_(mg)' in available_nutrients else -1
                        if iron_idx >= 0:
                            member_target[iron_idx] *= 1.3
                        if vitc_idx >= 0:
                            member_target[vitc_idx] *= 1.3
        
        aggregate_target += member_target * ratio
        total_weight += ratio
        total_tee += tee * ratio
        total_calories += tee * 0.2 * ratio  # 20% from vegetables
        
        conditions.update(p.get('conditions', []))
        allergies.update(p.get('allergies', []))
    
    if total_weight > 0:
        aggregate_target /= total_weight
        total_tee /= total_weight
    
    # Generate household notes
    if 'Anemia' in conditions:
        household_notes.append("⚠️ Increased iron needs for anemia")
    if 'Hypertension' in conditions:
        household_notes.append("❤️ Prioritizing potassium-rich vegetables")
    if 'Diabetes' in conditions:
        household_notes.append("🩸 Lower glycemic vegetables recommended")
    
    return aggregate_target, {
        'total_weight': total_weight,
        'total_tee': total_tee,
        'total_calories': total_calories,
        'conditions': list(conditions),
        'allergies': list(allergies),
        'household_notes': household_notes,
        'family_size': len(profiles)
    }

def generate_recommendations_with_ml(agg_target, agg_info, district, use_seasonal_filter=True):
    """Generate recommendations using available ML models or fallback"""
    # Filter vegetables
    current_month = datetime.now().month
    filtered_veg = veggies.copy()
    
    # 1. Apply allergy filters
    if agg_info['allergies']:
        for allergy in agg_info['allergies']:
            if allergy in ALLERGY_MAPPINGS:
                for keyword in ALLERGY_MAPPINGS[allergy]:
                    filtered_veg = filtered_veg[~filtered_veg['Shrt_Desc'].str.contains(keyword, case=False, na=False)]
    
    # 2. Apply medical condition filters
    if agg_info['conditions']:
        for condition in agg_info['conditions']:
            if condition in MEDICAL_CONDITIONS:
                for avoid in MEDICAL_CONDITIONS[condition]['avoid']:
                    if avoid == 'HIGH SODIUM':
                        filtered_veg = filtered_veg[filtered_veg['Sodium_(mg)'] < 100]
                    else:
                        filtered_veg = filtered_veg[~filtered_veg['Shrt_Desc'].str.contains(avoid, case=False, na=False)]
    
    # 3. Apply seasonal filter
    if use_seasonal_filter:
        seasonal_indices = []
        for idx, row in filtered_veg.iterrows():
            season_status = is_in_season(row['Shrt_Desc'], district, current_month)
            if season_status in ['Maha Season', 'Yala Season', None]:
                seasonal_indices.append(idx)
        filtered_veg = filtered_veg.loc[seasonal_indices]
    
    if filtered_veg.empty:
        return pd.DataFrame()  # Return empty if no vegetables pass filters
    
    # Try to use XGBoost if available
    if 'xgboost' in st.session_state.models and st.session_state.models['xgboost'] is not None:
        try:
            # Prepare features
            avg_age = np.mean([p['age'] for p in st.session_state.profiles]) if st.session_state.profiles else 30
            avg_bmi = np.mean([p['bmi'] for p in st.session_state.profiles]) if st.session_state.profiles else 22
            avg_tee = agg_info['total_tee']
            
            # Prepare feature matrix
            X_test = []
            for _, veg_row in filtered_veg.iterrows():
                # User features + vegetable features
                user_features = np.array([avg_age, avg_bmi, avg_tee])
                veg_features = veg_row[available_nutrients].values
                combined_features = np.concatenate([user_features, veg_features])
                X_test.append(combined_features)
            
            # Convert to DMatrix
            dtest = xgb.DMatrix(np.array(X_test))
            
            # Predict
            scores = st.session_state.models['xgboost'].predict(dtest)
            filtered_veg['ml_score'] = scores
            
        except Exception as e:
            st.warning(f"XGBoost prediction failed: {e}. Using fallback method.")
            # Fallback to similarity
            filtered_veg = generate_recommendations_fallback(agg_target, agg_info, district, filtered_veg)
            return filtered_veg.head(7)
    else:
        # Use fallback method
        filtered_veg = generate_recommendations_fallback(agg_target, agg_info, district, filtered_veg)
        return filtered_veg.head(7)
    
    # Add SPN confidence if available
    if 'spn' in st.session_state.models and st.session_state.models['spn'] is not None:
        try:
            # Simulate SPN confidence (replace with actual SPN call)
            np.random.seed(42)
            filtered_veg['spn_confidence'] = np.random.uniform(0.6, 0.95, len(filtered_veg))
            filtered_veg['combined_score'] = filtered_veg['ml_score'] * filtered_veg['spn_confidence']
        except:
            filtered_veg['spn_confidence'] = 0.8
            filtered_veg['combined_score'] = filtered_veg['ml_score']
    else:
        filtered_veg['spn_confidence'] = 0.8
        filtered_veg['combined_score'] = filtered_veg['ml_score']
    
    # Add additional information
    filtered_veg['season'] = filtered_veg['Shrt_Desc'].apply(
        lambda x: is_in_season(x, district, current_month)
    )
    
    # Mark CotD priority vegetables
    filtered_veg['cotd_priority'] = False
    for idx in filtered_veg.index:
        veg_name = filtered_veg.loc[idx, 'Shrt_Desc']
        for usda_name, local_name in USDA_TO_COTD_MAP.items():
            if usda_name in veg_name.upper():
                if local_name in COTD_VEGGIES.get(district, []):
                    filtered_veg.loc[idx, 'cotd_priority'] = True
    
    return filtered_veg.sort_values('combined_score', ascending=False).head(7)

def generate_recommendations_fallback(agg_target, agg_info, district, filtered_veg):
    """Fallback recommendation method using cosine similarity"""
    # Calculate similarity
    scaler = StandardScaler()
    veg_matrix = scaler.fit_transform(filtered_veg[available_nutrients])
    
    # Adjust target for medical conditions
    adjusted_target = agg_target.copy()
    if 'Anemia' in agg_info['conditions']:
        iron_idx = available_nutrients.index('Iron_(mg)') if 'Iron_(mg)' in available_nutrients else -1
        if iron_idx >= 0:
            adjusted_target[iron_idx] *= 2.0
    if 'Hypertension' in agg_info['conditions']:
        potassium_idx = available_nutrients.index('Potassium_(mg)') if 'Potassium_(mg)' in available_nutrients else -1
        if potassium_idx >= 0:
            adjusted_target[potassium_idx] *= 1.5
    
    # Cosine similarity
    sims = cosine_similarity(adjusted_target.reshape(1, -1), veg_matrix)[0]
    
    # Add CotD priority boost
    for i, idx in enumerate(filtered_veg.index):
        veg_name = filtered_veg.loc[idx, 'Shrt_Desc']
        for usda_name, local_name in USDA_TO_COTD_MAP.items():
            if usda_name in veg_name.upper():
                if local_name in COTD_VEGGIES.get(district, []):
                    sims[i] += 0.3  # Boost for CotD priority
    
    filtered_veg['similarity_score'] = sims
    filtered_veg['spn_confidence'] = np.random.uniform(0.6, 0.95, len(filtered_veg))  # Simulated SPN
    filtered_veg['combined_score'] = filtered_veg['similarity_score'] * filtered_veg['spn_confidence']
    
    # Add season info
    current_month = datetime.now().month
    filtered_veg['season'] = filtered_veg['Shrt_Desc'].apply(
        lambda x: is_in_season(x, district, current_month)
    )
    
    # Mark CotD priority
    filtered_veg['cotd_priority'] = False
    for idx in filtered_veg.index:
        veg_name = filtered_veg.loc[idx, 'Shrt_Desc']
        for usda_name, local_name in USDA_TO_COTD_MAP.items():
            if usda_name in veg_name.upper():
                if local_name in COTD_VEGGIES.get(district, []):
                    filtered_veg.loc[idx, 'cotd_priority'] = True
    
    return filtered_veg.sort_values('combined_score', ascending=False)

def recommend_recipes(vegetables, agg_info, district, num_days=7):
    """Recommend recipes and create weekly plan"""
    matched_recipes = []
    
    # Clean recipe vegetables
    recipes_clean = recipes.copy()
    recipes_clean['main_veg_clean'] = recipes_clean['vegetables_usda'].apply(
        lambda x: [clean_string_for_matching(v) for v in ast.literal_eval(x)] if isinstance(x, str) else []
    )
    
    # Get top 3 vegetables for recipe matching
    if vegetables.empty:
        return [], []
    
    top_vegetables = vegetables.head(3)['Shrt_Desc'].tolist()
    
    for veg_name in top_vegetables:
        veg_clean = clean_string_for_matching(veg_name)
        
        # Find recipes containing this vegetable
        candidates = recipes_clean[
            recipes_clean['main_veg_clean'].apply(
                lambda x: veg_clean in x
            )
        ]
        
        if not candidates.empty:
            # Filter for traditional Sri Lankan recipes
            traditional_candidates = candidates[
                candidates['cuisine_type'].str.contains('Traditional|Sri Lankan', case=False, na=False)
            ]
            
            if not traditional_candidates.empty:
                candidates = traditional_candidates
            
            # Take top 2 recipes per vegetable
            for _, rec in candidates.head(2).iterrows():
                # Scale for household
                scaled_servings = rec['servings'] * agg_info['total_weight']
                total_cost = rec['cost_per_serving_lkr'] * scaled_servings
                
                # Apply southern district discount
                if district in ['Hambantota', 'Matara', 'Galle']:
                    total_cost *= 0.95
                
                matched_recipes.append({
                    'recipe_name': rec['recipe_name'],
                    'scaled_servings': round(scaled_servings),
                    'total_cost': round(total_cost, 2),
                    'ingredients': rec['other_ingredients'],
                    'scaled_ingredients': scale_ingredients(rec['other_ingredients'], agg_info['total_weight']),
                    'prep_time': rec.get('prep_time_minutes', 'N/A'),
                    'cooking_time': rec.get('cooking_time_minutes', 'N/A'),
                    'main_vegetable': veg_name.split(',')[0],
                    'recipe_id': rec.get('recipe_id', len(matched_recipes))
                })
    
    # Remove duplicates
    unique_recipes = {}
    for recipe in matched_recipes:
        if recipe['recipe_name'] not in unique_recipes:
            unique_recipes[recipe['recipe_name']] = recipe
    
    # Create weekly plan
    weekly_plan = create_weekly_plan(list(unique_recipes.values())[:5], vegetables, num_days)
    
    return list(unique_recipes.values())[:5], weekly_plan

def create_weekly_plan(recipes, vegetables, num_days=7):
    """Create a weekly meal rotation plan"""
    if not recipes or vegetables.empty:
        return []
    
    plan = []
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    for i in range(min(num_days, 7)):
        recipe_idx = i % len(recipes)
        veg_idx = i % min(len(vegetables), 5)  # Use top 5 vegetables
        
        day_plan = {
            'day': days[i],
            'recipe': recipes[recipe_idx]['recipe_name'],
            'main_vegetable': vegetables.iloc[veg_idx]['Shrt_Desc'].split(',')[0],
            'focus_nutrient': get_focus_nutrient(vegetables.iloc[veg_idx]),
            'estimated_cost': recipes[recipe_idx]['total_cost'] / 7,  # Daily portion
            'servings': recipes[recipe_idx]['scaled_servings']
        }
        plan.append(day_plan)
    
    return plan

def get_focus_nutrient(vegetable_row):
    """Get the most abundant nutrient for a vegetable"""
    nutrient_cols = ['Protein_(g)', 'Iron_(mg)', 'Vit_C_(mg)', 'Fiber_TD_(g)', 'Calcium_(mg)']
    max_nutrient = ''
    max_value = 0
    
    for col in nutrient_cols:
        if col in vegetable_row and vegetable_row[col] > max_value:
            max_value = vegetable_row[col]
            max_nutrient = col.replace('_', ' ').replace('(g)', '').replace('(mg)', '')
    
    return max_nutrient if max_nutrient else 'Mixed Nutrients'

def generate_shopping_list(recipes, weekly_plan, agg_info):
    """Generate consolidated shopping list from recipes"""
    if not recipes:
        return []
    
    shopping_items = {}
    
    for recipe in recipes:
        if pd.isna(recipe['ingredients']):
            continue
            
        ingredients = str(recipe['ingredients']).split(',')
        for item in ingredients:
            item = item.strip()
            if not item:
                continue
                
            # Parse quantity and ingredient
            import re
            match = re.match(r'^(\d*\.?\d+)\s*([a-zA-Z]*)\s*(.*)', item)
            if match:
                quantity = float(match.group(1))
                unit = match.group(2).lower()
                ingredient_name = match.group(3).strip().lower()
                
                # Scale for weekly plan (assuming recipe used once in week)
                scaled_quantity = quantity * agg_info['total_weight']
                
                key = f"{ingredient_name}_{unit}"
                if key in shopping_items:
                    shopping_items[key]['quantity'] += scaled_quantity
                else:
                    shopping_items[key] = {
                        'ingredient': ingredient_name.title(),
                        'quantity': scaled_quantity,
                        'unit': unit
                    }
    
    # Convert to list and sort
    shopping_list = []
    for item in shopping_items.values():
        # Convert units for readability
        if item['unit'] == 'g' and item['quantity'] >= 1000:
            item['quantity'] /= 1000
            item['unit'] = 'kg'
        elif item['unit'] == 'ml' and item['quantity'] >= 1000:
            item['quantity'] /= 1000
            item['unit'] = 'l'
        
        shopping_list.append(item)
    
    return shopping_list

# --- 5. TRANSLATION FUNCTIONS ---
def translate_text(text, lang):
    """Simple translation function"""
    translations = {
        'Vegetables': {'Sinhala': 'පලාවල්', 'Tamil': 'காய்கறிகள்'},
        'Recipes': {'Sinhala': 'වට්ටෝරු', 'Tamil': 'சமையல் வகைகள்'},
        'Household': {'Sinhala': 'ගෘහස්ථ', 'Tamil': 'வீட்டு'},
        'Generate Plan': {'Sinhala': 'සැලැස්ම තනන්න', 'Tamil': 'திட்டத்தை உருவாக்கு'},
        'Family Member': {'Sinhala': 'පවුලේ සාමාජිකයා', 'Tamil': 'குடும்ப உறுப்பினர்'},
        'Save': {'Sinhala': 'සුරකින්න', 'Tamil': 'சேமி'},
        'Load': {'Sinhala': 'පූරණය කරන්න', 'Tamil': 'ஏற்று'},
        'Settings': {'Sinhala': 'සැකසුම්', 'Tamil': 'அமைப்புகள்'},
        'Language': {'Sinhala': 'භාෂාව', 'Tamil': 'மொழி'},
        'District': {'Sinhala': 'දිස්ත්‍රික්කය', 'Tamil': 'மாவட்டம்'},
        'Age': {'Sinhala': 'වයස', 'Tamil': 'வயது'},
        'Gender': {'Sinhala': 'ලිංගභේදය', 'Tamil': 'பாலினம்'},
        'Height': {'Sinhala': 'උස', 'Tamil': 'உயரம்'},
        'Weight': {'Sinhala': 'බර', 'Tamil': 'எடை'},
        'Activity Level': {'Sinhala': 'ක්‍රියාකාරී මට්ටම', 'Tamil': 'செயல்பாட்டு நிலை'},
        'Medical Conditions': {'Sinhala': 'දේවලුම් තත්වයන්', 'Tamil': 'மருத்துவ நிலைகள்'},
        'Allergies': {'Sinhala': 'අපසාමීයතා', 'Tamil': 'ஒவ்வாமை'},
        'Preferences': {'Sinhala': 'මනාප', 'Tamil': 'விருப்பங்கள்'},
        'Daily Budget': {'Sinhala': 'දිනපතා අයවැය', 'Tamil': 'தினசரி பட்ஜெட்'},
        'Seasonal Only': {'Sinhala': 'වාරෝත්පත්ති පමණි', 'Tamil': 'பருவமட்டும்'},
        'Sri Lanka Veggie Advisor (CotD 2024)': {
            'Sinhala': 'ශ්‍රී ලංකා පලා උපදේශක (තේ වසම සංගණනය 2024)',
            'Tamil': 'இலங்கை காய்கறி ஆலோசகர் (தேயிலை கள கணிப்பு 2024)'
        },
        'Personalized vegetable & recipe recommendations based on Census of Tea Domain 2024 guidelines': {
            'Sinhala': 'තේ වසම සංගණනය 2024 මාර්ගෝපදේශ අනුව පෞද්ගලිකරණය කරන ලද පලා හා වට්ටෝරු නිර්දේශ',
            'Tamil': 'தேயிலை கள கணிப்பு 2024 வழிகாட்டுதல்களின் அடிப்படையில் தனிப்பயனாக்கப்பட்ட காய்கறி மற்றும் சமையல் வகை பரிந்துரைகள்'
        },
        'Household Summary': {'Sinhala': 'ගෘහස්ථ සාරාංශය', 'Tamil': 'வீட்டு சுருக்கம்'},
        'Household Stats': {'Sinhala': 'ගෘහස්ථ සංඛ්‍යාලේඛන', 'Tamil': 'வீட்டு புள்ளிவிவரங்கள்'},
        'Family Size': {'Sinhala': 'පවුල් ප්‍රමාණය', 'Tamil': 'குடும்ப அளவு'},
        'Total TEE': {'Sinhala': 'මුළු TEE', 'Tamil': 'மொத்த TEE'},
        'Average BMI': {'Sinhala': 'සාමාන්‍ය BMI', 'Tamil': 'சராசரி BMI'},
        'Recommendation Settings': {'Sinhala': 'නිර්දේශ සැකසුම්', 'Tamil': 'பரிந்துரை அமைப்புகள்'},
        'Show only seasonal vegetables': {'Sinhala': 'වාරෝත්පත්ති පලා පමණි පෙන්වන්න', 'Tamil': 'பருவ காய்கறிகள் மட்டும் காட்டு'},
        'Filter to vegetables currently in season (Maha/Yala)': {
            'Sinhala': 'දැනට වාරෝත්පත්ති (මහ/යල) පලා වෙත පෙරහන් කරන්න',
            'Tamil': 'தற்போது பருவத்தில் உள்ள காய்கறிகளுக்கு வடிகட்டவும் (மகா/யல)'
        },
        'Max daily budget (LKR)': {'Sinhala': 'උපරිම දෛනික අයවැය (රු)', 'Tamil': 'அதிகபட்ச தினசரி பட்ஜெட் (ரூ)'},
        'Plan duration': {'Sinhala': 'සැලැස්ම කාල සීමාව', 'Tamil': 'திட்ட கால அளவு'},
        'Generate Personalized Plan': {'Sinhala': 'පෞද්ගලිකරණය කළ සැලැස්ම තනන්න', 'Tamil': 'தனிப்பயனாக்கப்பட்ட திட்டத்தை உருவாக்கு'},
        'Add family members or set your profile in the sidebar.': {
            'Sinhala': 'පැති තීරුවේ පවුලේ සාමාජිකයින් එක් කරන්න හෝ ඔබගේ පැතිකඩ සකසන්න.',
            'Tamil': 'பக்கப்பட்டியில் குடும்ப உறுப்பினர்களைச் சேர்க்கவும் அல்லது உங்கள் சுயவிவரத்தை அமைக்கவும்.'
        },
        'Affordability Analysis': {'Sinhala': 'කිරීමට හැකි විශ්ලේෂණය', 'Tamil': 'வாங்கும் திறன் பகுப்பாய்வு'},
        'Household CU': {'Sinhala': 'ගෘහස්ථ CU', 'Tamil': 'வீட்டு CU'},
        'Daily Cost': {'Sinhala': 'දෛනික පිරිවැය', 'Tamil': 'தினசரி செலவு'},
        'Affordability Gap': {'Sinhala': 'කිරීමට හැකි පරතරය', 'Tamil': 'வாங்கும் திறன் இடைவெளி'},
        'District Risk': {'Sinhala': 'දිස්ත්‍රික්ක අවදානම', 'Tamil': 'மாவட்ட ஆபத்து'},
        'Nutrient Coverage': {'Sinhala': 'පෝෂක ආවරණය', 'Tamil': 'ஊட்டச்சத்து உள்ளடக்கம்'},
        'Average Nutrients per Recommended Vegetable': {
            'Sinhala': 'නිර්දේශිත පලාවකට සාමාන්‍ය පෝෂක',
            'Tamil': 'பரிந்துரைக்கப்பட்ட காய்கறிக்கான சராசரி ஊட்டச்சத்துக்கள்'
        },
        'Nutrients': {'Sinhala': 'පෝෂක', 'Tamil': 'ஊட்டச்சத்துக்கள்'},
        'Amount': {'Sinhala': 'ප්‍රමාණය', 'Tamil': 'அளவு'},
        'Recommended Vegetables': {'Sinhala': 'නිර්දේශිත පලා', 'Tamil': 'பரிந்துரைக்கப்பட்ட காய்கறிகள்'},
        'No vegetables found matching your criteria. Try relaxing filters.': {
            'Sinhala': 'ඔබගේ නිර්ණායකවලට ගැලපෙන පලා හමු නොවීය. පෙරහන් ලිහිල් කිරීමට උත්සාහ කරන්න.',
            'Tamil': 'உங்கள் நிபந்தனைகளுக்கு பொருந்தக்கூடிய காய்கறிகள் கிடைக்கவில்லை. வடிப்பான்களை தளர்த்த முயற்சிக்கவும்.'
        },
        'Recommended Recipes': {'Sinhala': 'නිර්දේශිත වට්ටෝරු', 'Tamil': 'பரிந்துரைக்கப்பட்ட சமையல் வகைகள்'},
        'Ingredients for Household': {'Sinhala': 'ගෘහස්ථය සඳහා අමුද්‍රව්‍ය', 'Tamil': 'வீட்டிற்கான பொருட்கள்'},
        'Main Vegetable': {'Sinhala': 'ප්‍රධාන පලා', 'Tamil': 'முக்கிய காய்கறி'},
        'Cost per serving': {'Sinhala': 'පරිභෝජනයකට පිරිවැය', 'Tamil': 'ஒரு பரிமாற்றத்திற்கான செலவு'},
        'Prep time': {'Sinhala': 'සූදානම් කිරීමේ කාලය', 'Tamil': 'தயாரிப்பு நேரம்'},
        'Cooking time': {'Sinhala': 'හදා ගැනීමේ කාලය', 'Tamil': 'சமைக்கும் நேரம்'},
        'Weekly Meal Plan': {'Sinhala': 'සතිපතා ආහාර සැලැස්ම', 'Tamil': 'வாராந்திர உணவு திட்டம்'},
        'Shopping List': {'Sinhala': 'සාප්පු ලැයිස්තුව', 'Tamil': 'கடை பட்டியல்'},
        'Export & Feedback': {'Sinhala': 'අපනයනය සහ ප්‍රතිපෝෂණය', 'Tamil': 'ஏற்றுமதி & கருத்து'},
        'Was this plan useful?': {'Sinhala': 'මෙම සැලැස්ම ප්‍රයෝජනවත් වූයේද?', 'Tamil': 'இந்தத் திட்டம் பயனுள்ளதாக இருந்ததா?'},
        'Additional comments (optional)': {'Sinhala': 'අතිරේක අදහස් (විකල්ප)', 'Tamil': 'கூடுதல் கருத்துகள் (விருப்பத்தேர்வு)'},
        'Submit Feedback': {'Sinhala': 'ප්‍රතිපෝෂණය ඉදිරිපත් කරන්න', 'Tamil': 'கருத்தை சமர்ப்பிக்கவும்'},
        'For nutritional guidance only. Consult a healthcare professional for medical advice.': {
            'Sinhala': 'පෝෂණ මාර්ගෝපදේශය සඳහා පමණි. වෛද්‍ය උපදෙස් සඳහා සෞඛ්‍ය සේවා වෘත්තිකයෙකුගෙන් උපදෙස් ලබා ගන්න.',
            'Tamil': 'ஊட்டச்சத்து வழிகாட்டுதலுக்கு மட்டுமே. மருத்துவ ஆலோசனைக்கு ஒரு சுகாதார நிபுணரைக் கலந்தாலோசிக்கவும்.'
        }
    }
    
    if lang == 'English' or text not in translations:
        return text
    
    return translations[text].get(lang, text)

# --- 6. CUSTOM CSS FOR BETTER UI ---
def apply_custom_css():
    """Apply custom CSS for better UI"""
    css = """
    <style>
    :root {
        --primary: #27ae60;
        --primary-light: #2ecc71;
        --primary-dark: #219653;
        --secondary: #3498db;
        --bg-light: #f8f9fa;
        --bg-dark: #2c3e50;
        --text-light: #ecf0f1;
        --text-dark: #2c3e50;
        --warning: #f39c12;
        --danger: #e74c3c;
        --success: #2ecc71;
        --border-radius: 12px;
        --box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .stApp {
        background-color: var(--bg-light);
    }
    
    h1, h2, h3 {
        color: var(--primary-dark);
        font-weight: 700;
    }
    
    .stButton > button {
        background-color: var(--primary);
        color: white;
        border-radius: var(--border-radius);
        border: none;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: var(--primary-dark);
        transform: translateY(-2px);
        box-shadow: var(--box-shadow);
    }
    
    .metric-container {
        background: white;
        border-radius: var(--border-radius);
        box-shadow: var(--box-shadow);
        padding: 1.5rem;
        border-left: 5px solid var(--primary-light);
        margin-bottom: 1rem;
    }
    
    .cotd-priority {
        background: linear-gradient(135deg, #fff3cd, #ffeaa7);
        color: #856404;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-right: 8px;
        margin-bottom: 8px;
        border: 1px solid #f1c40f;
    }
    
    .season-badge {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        color: #155724;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-right: 8px;
        margin-bottom: 8px;
        border: 1px solid #28a745;
    }
    
    .spn-confidence {
        background: linear-gradient(135deg, #d1ecf1, #bee5eb);
        color: #0c5460;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-right: 8px;
        margin-bottom: 8px;
        border: 1px solid #17a2b8;
    }
    
    .vegetable-card {
        background: white;
        border-radius: var(--border-radius);
        box-shadow: var(--box-shadow);
        padding: 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid #e9ecef;
        transition: all 0.3s ease;
    }
    
    .vegetable-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
    }
    
    .recipe-card {
        background: white;
        border-radius: var(--border-radius);
        box-shadow: var(--box-shadow);
        padding: 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid #e9ecef;
    }
    
    .shopping-item {
        background: white;
        border-radius: var(--border-radius);
        box-shadow: var(--box-shadow);
        padding: 1rem;
        margin-bottom: 0.5rem;
        border-left: 4px solid var(--primary);
    }
    
    .day-plan {
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
        border-radius: var(--border-radius);
        padding: 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid #90caf9;
    }
    
    .affordability-good {
        color: var(--success);
        font-weight: 700;
    }
    
    .affordability-warning {
        color: var(--warning);
        font-weight: 700;
    }
    
    .affordability-bad {
        color: var(--danger);
        font-weight: 700;
    }
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .stButton > button {
            width: 100%;
            margin-bottom: 0.5rem;
        }
        
        .metric-container {
            margin-bottom: 0.5rem;
        }
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# --- 7. SIDEBAR COMPONENTS ---
def render_sidebar():
    """Render the sidebar with all controls"""
    st.sidebar.title("🌍 " + translate_text("Settings", st.session_state.language))
    
    # Language selection
    lang = st.sidebar.selectbox(
        translate_text("Language", st.session_state.language),
        ['English', 'Sinhala', 'Tamil'],
        key='language_select'
    )
    st.session_state.language = lang
    
    # Dark mode toggle
    dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
    if dark_mode != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_mode
        st.rerun()
    
    # District selection
    district = st.sidebar.selectbox(
        "📍 " + translate_text("District", lang),
        SRI_LANKA_DISTRICTS,
        index=SRI_LANKA_DISTRICTS.index('Hambantota') if 'Hambantota' in SRI_LANKA_DISTRICTS else 0
    )
    
    # Load ML models button
    if not st.session_state.models_loaded:
        if st.sidebar.button("🤖 Load ML Models", use_container_width=True):
            with st.spinner("Loading ML models..."):
                load_ml_models()
                st.success("ML models loaded!")
                st.rerun()
    
    # Planning mode
    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 " + translate_text("Planning Mode", lang))
    planning_mode = st.sidebar.radio(
        "",
        ["👨‍👩‍👧‍👦 " + translate_text("Family", lang), "👤 " + translate_text("Individual", lang)],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    # Save/Load section
    st.sidebar.markdown("---")
    st.sidebar.subheader("💾 " + translate_text("Save/Load", lang))
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        save_name = st.text_input(
            translate_text("Save as", lang),
            placeholder=translate_text("Family name", lang),
            key="save_name_input"
        )
    
    if st.sidebar.button("💾 " + translate_text("Save Household", lang), use_container_width=True):
        if save_name and st.session_state.profiles:
            st.session_state.saved_households[save_name] = {
                'profiles': [p.copy() for p in st.session_state.profiles],
                'district': district,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M")
            }
            st.sidebar.success(f"✅ Saved '{save_name}'")
        else:
            st.sidebar.warning("Please add profiles first")
    
    if st.session_state.saved_households:
        load_choice = st.sidebar.selectbox(
            translate_text("Load Household", lang),
            list(st.session_state.saved_households.keys()),
            key="load_select"
        )
        
        if st.sidebar.button("📂 " + translate_text("Load Selected", lang), use_container_width=True):
            loaded = st.session_state.saved_households[load_choice]
            st.session_state.profiles = [p.copy() for p in loaded['profiles']]
            st.sidebar.success(f"✅ Loaded '{load_choice}'")
            st.rerun()
    
    # Profile management
    st.sidebar.markdown("---")
    st.sidebar.subheader("👥 " + translate_text("Household Manager", lang))
    
    if planning_mode.startswith("👨‍👩‍👧‍👦"):
        render_family_mode(lang)
    else:
        render_individual_mode(lang)
    
    return district, planning_mode

def render_family_mode(lang):
    """Render family mode interface"""
    with st.sidebar.expander("➕ " + translate_text("Add Family Member", lang), expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input(translate_text("Name", lang), key="name_input")
            age = st.number_input(translate_text("Age", lang), 1, 100, 30, key="age_input")
            gender = st.selectbox(
                translate_text("Gender", lang),
                ["Male", "Female", "Prefer not to say"],
                key="gender_input"
            )
        
        with col2:
            height = st.number_input(translate_text("Height (cm)", lang), 50, 250, 165, key="height_input")
            weight = st.number_input(translate_text("Weight (kg)", lang), 10, 200, 60, key="weight_input")
            activity = st.selectbox(
                translate_text("Activity Level", lang),
                list(ACTIVITY_MULTIPLIERS.keys()),
                index=2,
                key="activity_input"
            )
        
        # Medical conditions
        conditions = st.multiselect(
            translate_text("Medical Conditions", lang),
            list(MEDICAL_CONDITIONS.keys()),
            key="conditions_input"
        )
        
        # Allergies
        allergies = st.multiselect(
            translate_text("Allergies/Intolerances", lang),
            list(ALLERGY_MAPPINGS.keys()),
            key="allergies_input"
        )
        
        # Preferences
        preferences = st.text_area(
            translate_text("Food Preferences", lang),
            placeholder=translate_text("e.g., 'no bitter gourd', 'prefer leafy greens'", lang),
            key="preferences_input"
        )
        
        col_add, _ = st.columns([2, 1])
        with col_add:
            if st.button("➕ " + translate_text("Add to Household", lang), use_container_width=True):
                if name:
                    bmr = calculate_bmr(weight, height, age, gender)
                    tee = calculate_tee(bmr, activity)
                    bmi = weight / ((height/100) ** 2)
                    
                    profile = {
                        'name': name,
                        'age': age,
                        'gender': gender,
                        'height': height,
                        'weight': weight,
                        'bmi': round(bmi, 1),
                        'bmr': round(bmr, 1),
                        'tee': round(tee, 1),
                        'activity': activity,
                        'conditions': conditions,
                        'allergies': allergies,
                        'preferences': preferences,
                        'role': determine_cotd_role(age, gender)[0]
                    }
                    
                    st.session_state.profiles.append(profile)
                    st.success(f"✅ Added {name} to household")
                    st.rerun()
                else:
                    st.error("Please enter a name")
    
    # Display current household
    if st.session_state.profiles:
        st.sidebar.markdown("### " + translate_text("Current Household", lang))
        for i, p in enumerate(st.session_state.profiles):
            cols = st.sidebar.columns([3, 1])
            with cols[0]:
                role_icon = "👶" if p['age'] < 8 else "👧" if p['age'] < 16 else "🧑" if p['age'] < 60 else "🧓"
                st.write(f"{role_icon} **{p['name']}** ({p['age']}y, {p['gender'][0]})")
                st.caption(f"{p['role']} | BMI: {p['bmi']} | TEE: {p['tee']:.0f} kcal")
            with cols[1]:
                if st.button("🗑️", key=f"del_{i}"):
                    st.session_state.profiles.pop(i)
                    st.rerun()

def render_individual_mode(lang):
    """Render individual mode interface"""
    with st.sidebar.expander("👤 " + translate_text("Individual Profile", lang), expanded=True):
        name = st.text_input(translate_text("Name", lang), value="Individual", key="ind_name")
        age = st.number_input(translate_text("Age", lang), 1, 100, 30, key="ind_age")
        gender = st.selectbox(
            translate_text("Gender", lang),
            ["Male", "Female", "Prefer not to say"],
            key="ind_gender"
        )
        height = st.number_input(translate_text("Height (cm)", lang), 50, 250, 165, key="ind_height")
        weight = st.number_input(translate_text("Weight (kg)", lang), 10, 200, 60, key="ind_weight")
        activity = st.selectbox(
            translate_text("Activity Level", lang),
            list(ACTIVITY_MULTIPLIERS.keys()),
            index=2,
            key="ind_activity"
        )
        
        conditions = st.multiselect(
            translate_text("Medical Conditions", lang),
            list(MEDICAL_CONDITIONS.keys()),
            key="ind_conditions"
        )
        
        allergies = st.multiselect(
            translate_text("Allergies/Intolerances", lang),
            list(ALLERGY_MAPPINGS.keys()),
            key="ind_allergies"
        )
        
        preferences = st.text_area(
            translate_text("Food Preferences", lang),
            key="ind_preferences"
        )
        
        if st.button("✅ " + translate_text("Set Profile", lang), use_container_width=True):
            bmr = calculate_bmr(weight, height, age, gender)
            tee = calculate_tee(bmr, activity)
            bmi = weight / ((height/100) ** 2)
            
            profile = {
                'name': name,
                'age': age,
                'gender': gender,
                'height': height,
                'weight': weight,
                'bmi': round(bmi, 1),
                'bmr': round(bmr, 1),
                'tee': round(tee, 1),
                'activity': activity,
                'conditions': conditions,
                'allergies': allergies,
                'preferences': preferences,
                'role': determine_cotd_role(age, gender)[0]
            }
            
            st.session_state.profiles = [profile]
            st.success(f"✅ Profile set for {name}")
            st.rerun()

# --- 8. MAIN APP LAYOUT ---
def main():
    # Apply custom CSS
    apply_custom_css()
    
    # Header
    st.title("🌱 " + translate_text("Sri Lanka Veggie Advisor (CotD 2024)", st.session_state.language))
    st.markdown(f"*{translate_text('Personalized vegetable & recipe recommendations based on Census of Tea Domain 2024 guidelines', st.session_state.language)}*")
    
    # Get settings from sidebar
    district, planning_mode = render_sidebar()
    
    # Main content area in two columns
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.subheader("👥 " + translate_text("Household Summary", st.session_state.language))
        
        if st.session_state.profiles:
            # Display household summary
            total_tee = sum([p['tee'] for p in st.session_state.profiles])
            avg_bmi = np.mean([p['bmi'] for p in st.session_state.profiles])
            
            st.markdown(f"""
            <div class="metric-container">
                <h4>📊 {translate_text('Household Stats', st.session_state.language)}</h4>
                <p>👥 {translate_text('Family Size', st.session_state.language)}: {len(st.session_state.profiles)}</p>
                <p>⚡ {translate_text('Total TEE', st.session_state.language)}: {total_tee:.0f} kcal/day</p>
                <p>⚖️ {translate_text('Average BMI', st.session_state.language)}: {avg_bmi:.1f}</p>
                <p>📍 {translate_text('District', st.session_state.language)}: {district}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display household notes if any
            if st.session_state.last_result and 'aggregate_info' in st.session_state.last_result:
                agg_info = st.session_state.last_result['aggregate_info']
                if agg_info.get('household_notes'):
                    with st.expander("📝 Household Notes"):
                        for note in agg_info['household_notes']:
                            st.write(f"• {note}")
        
        else:
            st.info("👈 " + translate_text("Add family members or set your profile in the sidebar.", st.session_state.language))
        
        # Controls
        st.markdown("---")
        st.subheader("🎯 " + translate_text("Recommendation Settings", st.session_state.language))
        
        use_seasonal = st.checkbox(
            "🌿 " + translate_text("Show only seasonal vegetables", st.session_state.language),
            value=True,
            help=translate_text("Filter to vegetables currently in season (Maha/Yala)", st.session_state.language)
        )
        
        max_daily_cost = st.number_input(
            "💰 " + translate_text("Max daily budget (LKR)", st.session_state.language),
            min_value=100,
            max_value=5000,
            value=1000,
            step=100
        )
        
        plan_days = st.selectbox(
            "📅 " + translate_text("Plan duration", st.session_state.language),
            [7, 14, 30],
            index=0
        )
        
        if st.button("🚀 " + translate_text("Generate Personalized Plan", st.session_state.language), 
                    type="primary", use_container_width=True):
            if not st.session_state.profiles:
                st.error("Please add at least one family member first")
            else:
                with st.spinner(translate_text("Generating personalized recommendations...", st.session_state.language)):
                    # Aggregate household
                    agg_target, agg_info = aggregate_household(st.session_state.profiles)
                    
                    # Generate recommendations
                    recommendations = generate_recommendations_with_ml(
                        agg_target, agg_info, district, use_seasonal_filter=use_seasonal
                    )
                    
                    # Get recipes and weekly plan
                    recipes_list, weekly_plan = recommend_recipes(recommendations, agg_info, district, plan_days)
                    
                    # Generate shopping list
                    shopping_list = generate_shopping_list(recipes_list, weekly_plan, agg_info)
                    
                    # Calculate costs
                    base_daily_cost = 905  # LKR base cost per CU
                    daily_cost = base_daily_cost * agg_info['total_weight']
                    risk_factor = DISTRICT_RISKS.get(district, 0.37)
                    affordability_gap = daily_cost * risk_factor
                    
                    # Store results
                    st.session_state.last_result = {
                        'recommendations': recommendations,
                        'recipes': recipes_list,
                        'weekly_plan': weekly_plan,
                        'shopping_list': shopping_list,
                        'aggregate_info': agg_info,
                        'district': district,
                        'settings': {
                            'use_seasonal': use_seasonal,
                            'max_daily_cost': max_daily_cost,
                            'plan_days': plan_days
                        },
                        'cost_info': {
                            'daily_cost': daily_cost,
                            'affordability_gap': affordability_gap,
                            'risk_factor': risk_factor,
                            'total_weight': agg_info['total_weight']
                        }
                    }
                    st.success("✅ Personalized plan generated successfully!")
    
    with col2:
        if st.session_state.last_result:
            res = st.session_state.last_result
            
            # Affordability Metrics
            st.subheader("💰 " + translate_text("Affordability Analysis", st.session_state.language))
            
            cols = st.columns(4)
            with cols[0]:
                st.metric(
                    translate_text("Household CU", st.session_state.language),
                    f"{res['cost_info']['total_weight']:.2f}"
                )
            with cols[1]:
                st.metric(
                    translate_text("Daily Cost", st.session_state.language),
                    f"LKR {res['cost_info']['daily_cost']:.0f}"
                )
            with cols[2]:
                gap_percent = (res['cost_info']['affordability_gap'] / res['cost_info']['daily_cost']) * 100
                gap_class = "affordability-good" if gap_percent < 20 else "affordability-warning" if gap_percent < 40 else "affordability-bad"
                st.markdown(f"""
                <div class="metric-container">
                    <h4>{translate_text('Affordability Gap', st.session_state.language)}</h4>
                    <p class="{gap_class}">LKR {res['cost_info']['affordability_gap']:.0f}</p>
                    <small>{gap_percent:.1f}% of daily cost</small>
                </div>
                """, unsafe_allow_html=True)
            with cols[3]:
                st.metric(
                    translate_text("District Risk", st.session_state.language),
                    f"{res['cost_info']['risk_factor']*100:.0f}%"
                )
            
            # CotD priority note
            cotd_priority_veggies = COTD_VEGGIES.get(district, [])
            if cotd_priority_veggies:
                st.info(f"**🏆 CotD 2024 Priority for {district}:** {', '.join(cotd_priority_veggies[:3])}")
            
            # Nutrient visualization
            st.markdown("---")
            st.subheader("📊 " + translate_text("Nutrient Coverage", st.session_state.language))
            
            if not res['recommendations'].empty:
                veg_nutrients = res['recommendations'][available_nutrients].mean()
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=[n.replace('_', ' ').replace('(g)', '').replace('(mg)', '') for n in available_nutrients],
                    y=veg_nutrients.values,
                    name='Provided by Vegetables',
                    marker_color='#2ecc71',
                    text=veg_nutrients.values.round(1),
                    textposition='auto'
                ))
                
                fig.update_layout(
                    title=translate_text('Average Nutrients per Recommended Vegetable', st.session_state.language),
                    xaxis_title=translate_text('Nutrients', st.session_state.language),
                    yaxis_title=translate_text('Amount', st.session_state.language),
                    height=350,
                    showlegend=False,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Main recommendations section (full width)
    if st.session_state.last_result:
        res = st.session_state.last_result
        
        st.markdown("---")
        st.subheader("🥦 " + translate_text("Recommended Vegetables", st.session_state.language))
        
        if res['recommendations'].empty:
            st.warning(translate_text("No vegetables found matching your criteria. Try relaxing filters.", st.session_state.language))
        else:
            # Create cards for each vegetable
            for idx, row in res['recommendations'].iterrows():
                veg_name = row['Shrt_Desc'].split(',')[0]
                local_name = row.get('Sinhala_Name', '') if st.session_state.language == 'Sinhala' else row.get('Tamil_Name', '') if st.session_state.language == 'Tamil' else veg_name
                
                st.markdown(f"""
                <div class="vegetable-card">
                    <h4>{veg_name}</h4>
                    <small>{local_name}</small>
                    <div style="margin-top: 10px;">
                """, unsafe_allow_html=True)
                
                # Badges
                col_badges = st.columns(4)
                with col_badges[0]:
                    if row.get('cotd_priority'):
                        st.markdown('<span class="cotd-priority">🏆 CotD Priority</span>', unsafe_allow_html=True)
                with col_badges[1]:
                    if row.get('season') and row['season'] != 'Off Season':
                        st.markdown(f'<span class="season-badge">🌱 {row["season"]}</span>', unsafe_allow_html=True)
                with col_badges[2]:
                    if 'spn_confidence' in row:
                        confidence_percent = row['spn_confidence'] * 100
                        st.markdown(f'<span class="spn-confidence">🤖 SPN: {confidence_percent:.0f}%</span>', unsafe_allow_html=True)
                with col_badges[3]:
                    if 'ml_score' in row:
                        score_percent = (row['ml_score'] / row['ml_score'].max()) * 100 if row['ml_score'].max() > 0 else 0
                        st.markdown(f'<span class="cotd-priority">🎯 ML: {score_percent:.0f}%</span>', unsafe_allow_html=True)
                    elif 'similarity_score' in row:
                        score_percent = (row['similarity_score'] / row['similarity_score'].max()) * 100 if row['similarity_score'].max() > 0 else 0
                        st.markdown(f'<span class="cotd-priority">🎯 Similarity: {score_percent:.0f}%</span>', unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Nutrients in columns
                nutrient_cols = st.columns(4)
                nutrient_display = {
                    'Protein_(g)': 'Protein',
                    'Iron_(mg)': 'Iron',
                    'Vit_C_(mg)': 'Vitamin C',
                    'Fiber_TD_(g)': 'Fiber',
                    'Calcium_(mg)': 'Calcium',
                    'Potassium_(mg)': 'Potassium'
                }
                
                for i, (nutrient_key, nutrient_name) in enumerate(nutrient_display.items()):
                    if nutrient_key in row and row[nutrient_key] > 0:
                        col_idx = i % 4
                        with nutrient_cols[col_idx]:
                            st.metric(nutrient_name, f"{row[nutrient_key]:.1f}")
                
                st.markdown("</div>", unsafe_allow_html=True)
        
        # Recommended Recipes
        st.markdown("---")
        st.subheader("🍛 " + translate_text("Recommended Recipes", st.session_state.language))
        
        if res['recipes']:
            for i, recipe in enumerate(res['recipes']):
                with st.expander(f"📖 **{recipe['recipe_name']}** • LKR {recipe['total_cost']} • {recipe['scaled_servings']} servings", 
                               expanded=i == 0):
                    col_rec1, col_rec2 = st.columns([2, 1])
                    
                    with col_rec1:
                        st.markdown("**" + translate_text("Ingredients for Household", st.session_state.language) + ":**")
                        st.write(recipe['scaled_ingredients'])
                        
                        st.markdown("**" + translate_text("Main Vegetable", st.session_state.language) + ":**")
                        st.info(f"🥕 {recipe['main_vegetable']}")
                    
                    with col_rec2:
                        st.metric(
                            translate_text("Cost per serving", st.session_state.language),
                            f"LKR {recipe['total_cost']/recipe['scaled_servings']:.1f}"
                        )
                        
                        if recipe.get('prep_time') and recipe['prep_time'] != 'N/A':
                            st.metric(
                                translate_text("Prep time", st.session_state.language),
                                f"{recipe['prep_time']} min"
                            )
                        
                        if recipe.get('cooking_time') and recipe['cooking_time'] != 'N/A':
                            st.metric(
                                translate_text("Cooking time", st.session_state.language),
                                f"{recipe['cooking_time']} min"
                            )
        
        # Weekly Plan
        st.markdown("---")
        st.subheader("📅 " + translate_text("Weekly Meal Plan", st.session_state.language))
        
        if res['weekly_plan']:
            week_cols = st.columns(min(7, len(res['weekly_plan'])))
            
            for i, day_plan in enumerate(res['weekly_plan'][:7]):
                with week_cols[i]:
                    st.markdown(f"""
                    <div class="day-plan">
                        <h5>{day_plan['day']}</h5>
                        <p><strong>🍲 Recipe:</strong> {day_plan['recipe']}</p>
                        <p><strong>🥦 Main Veg:</strong> {day_plan['main_vegetable']}</p>
                        <p><strong>💪 Focus:</strong> {day_plan['focus_nutrient']}</p>
                        <p><strong>💰 Cost:</strong> LKR {day_plan['estimated_cost']:.0f}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Shopping List
        st.markdown("---")
        st.subheader("🛒 " + translate_text("Shopping List", st.session_state.language))
        
        if res['shopping_list']:
            total_cost = sum([recipe['total_cost'] for recipe in res['recipes']]) / res['settings']['plan_days'] if res['recipes'] else 0
            
            col_shop1, col_shop2 = st.columns([2, 1])
            with col_shop1:
                st.markdown(f"**Total estimated cost for {res['settings']['plan_days']} days:** LKR {total_cost:.0f}")
                
                for item in res['shopping_list']:
                    st.markdown(f"""
                    <div class="shopping-item">
                        <p><strong>{item['ingredient']}</strong>: {item['quantity']:.1f} {item['unit']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col_shop2:
                st.markdown("**📋 Tips:**")
                st.markdown("""
                - Buy seasonal vegetables for better prices
                - Visit local markets for fresh produce
                - Store leafy greens in airtight containers
                - Plan meals around market days
                """)
        
        # Export & Feedback
        st.markdown("---")
        st.subheader("📤 " + translate_text("Export & Feedback", st.session_state.language))
        
        col_exp1, col_exp2, col_exp3 = st.columns(3)
        
        with col_exp1:
            # Export as JSON
            export_data = {
                'district': district,
                'generated_date': datetime.now().strftime("%Y-%m-%d"),
                'household_size': res['aggregate_info']['family_size'],
                'recommended_vegetables': [
                    {
                        'name': row['Shrt_Desc'],
                        'local_name': row.get('Sinhala_Name', ''),
                        'key_nutrients': {n: row[n] for n in available_nutrients if n in row},
                        'season': row.get('season'),
                        'cotd_priority': row.get('cotd_priority', False),
                        'spn_confidence': float(row.get('spn_confidence', 0)) if 'spn_confidence' in row else None
                    }
                    for _, row in res['recommendations'].iterrows()
                ],
                'recipes': res['recipes'],
                'weekly_plan': res['weekly_plan'],
                'shopping_list': res['shopping_list'],
                'cost_analysis': res['cost_info']
            }
            
            st.download_button(
                label="📥 JSON Export",
                data=json.dumps(export_data, indent=2),
                file_name=f"veggie_plan_{district}_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col_exp2:
            # Export as CSV
            if not res['recommendations'].empty:
                csv_data = res['recommendations'][['Shrt_Desc'] + available_nutrients].to_csv(index=False)
                st.download_button(
                    label="📊 CSV Export",
                    data=csv_data,
                    file_name=f"vegetables_{district}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col_exp3:
            # Feedback
            st.markdown("**" + translate_text("Was this plan useful?", st.session_state.language) + "**")
            rating = st.select_slider(
                "",
                options=["", "⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"],
                label_visibility="collapsed",
                key="rating_slider"
            )
            
            if rating:
                feedback = st.text_area(
                    translate_text("Additional comments (optional)", st.session_state.language),
                    height=60,
                    key="feedback_text"
                )
                
                if st.button("📝 " + translate_text("Submit Feedback", st.session_state.language), 
                           use_container_width=True):
                    st.session_state.feedback_history.append({
                        'timestamp': datetime.now().isoformat(),
                        'district': district,
                        'rating': rating,
                        'feedback': feedback,
                        'profiles_count': len(st.session_state.profiles),
                        'plan_days': res['settings']['plan_days']
                    })
                    
                    # Show average rating if we have feedback
                    if st.session_state.feedback_history:
                        ratings = [f['rating'].count('⭐') for f in st.session_state.feedback_history]
                        avg_rating = sum(ratings) / len(ratings)
                        st.success(f"✅ Thank you! Average rating: {avg_rating:.1f}⭐")
                    else:
                        st.success("✅ Thank you for your feedback!")
    
    # Footer
    st.markdown("---")
    st.caption(
        f"🌱 **Sri Lanka Veggie Advisor** • "
        f"Census of Tea Domain 2024 Alignment • "
        f"{translate_text('For nutritional guidance only. Consult a healthcare professional for medical advice.', st.session_state.language)} • "
        f"© {datetime.now().year}"
    )

# --- 9. RUN THE APP ---
if __name__ == "__main__":
    main()
