"""
predict_future.py – Interactive model selection and retraining control.
Allows user to choose model type, checks if recently trained, and optionally retrains.
FIXED:
  - Use config for date splits.
  - Build climatology from training data only.
  - Pass climatology to feature engineers during training.
  - Add guard for tuning plots.
  - Use global drought default.
  - Date format changed to MM/DD/YYYY in output CSV.
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import os

sys.path.append(str(Path(__file__).parent))

from sklearn.preprocessing import LabelEncoder

from utils.config_loader import ConfigLoader
from utils.data_loader import DataLoader
from utils.target_creator import TargetCreator
from utils.feature_engineer import FeatureEngineer
from utils.api_client import OpenMeteoClient
from training.trainer import Trainer
from evaluation.comparator import ModelComparator
from prediction.future_predictor import FuturePredictor
from models.model_factory import ModelFactory

# Import visualization functions
from visualization.model_plots import plot_model_comparison, plot_overfitting_analysis
from visualization.learning_curves import plot_learning_curve, plot_cv_scores
from training.cross_validation import TimeSeriesCV


def _rename_collector_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names from the data collector to pipeline conventions."""
    mapping = {
        'consecutive_dry_days': 'cdd',
        'solar_radiation':      'solar_radi',
    }
    for old, new in mapping.items():
        if old in data.columns and new not in data.columns:
            data = data.rename(columns={old: new})
    return data


def get_last_training_time(model_type: str, base_dir: str = 'models') -> datetime | None:
    """
    Check modification times of both model variants for the given type.
    Returns the latest modification time if both exist, else None.
    """
    full_dir = Path(base_dir) / f'saved_future_full_{model_type}'
    future_dir = Path(base_dir) / f'saved_future_{model_type}'
    full_file = full_dir / f'{model_type}.pkl'
    future_file = future_dir / f'{model_type}.pkl'

    if full_file.exists() and future_file.exists():
        full_mtime = datetime.fromtimestamp(full_file.stat().st_mtime)
        future_mtime = datetime.fromtimestamp(future_file.stat().st_mtime)
        return max(full_mtime, future_mtime)
    return None


def build_climatology_from_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Build a climatology DataFrame (means and stds per city and day_of_year)
    from the given historical data. Uses the latest 20 years or all available.
    """
    df = data.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_year'] = df['date'].dt.dayofyear
    latest_year = df['date'].dt.year.max()
    baseline_start = latest_year - 20
    df_base = df[df['date'].dt.year >= baseline_start]
    vars_to_agg = ['tmax', 'tmin', 'precipitation', 'rh_mean']
    if 'surface_pressure' in df_base.columns:
        vars_to_agg.append('surface_pressure')
    agg_funcs = {v: ['mean', 'std'] for v in vars_to_agg}
    clim = (
        df_base.groupby(['city', 'day_of_year'])
               .agg(agg_funcs)
    )
    clim.columns = ['_'.join(col) for col in clim.columns]
    clim = clim.reset_index()
    for col in clim.columns:
        if col.endswith('_std'):
            clim[col] = clim[col].fillna(0.1).replace(0, 0.1)
    return clim


def train_models(model_type: str, config, train_data, val_data, label_encoder, class_names, train_climatology):
    """
    Train both full and future models for the given model type.
    Passes the climatology (built from training data) to feature engineers.
    Returns the trained models, feature engineers, and data splits.
    """
    print(f"\n{'='*70}")
    print(f"TRAINING PIPELINE  –  model_type='{model_type}'")
    print(f"{'='*70}")

    # --- Full feature model ---
    print(f"\n--- Training FULL feature model ({model_type}) ---")
    full_fe = FeatureEngineer(config, feature_set='full', climatology=train_climatology)   # pass climatology
    X_train_f = full_fe.prepare_features(
        train_data.drop(columns=['target_encoded']), fit=True
    )
    y_train_f = train_data['target_encoded'].values

    X_val_f = full_fe.prepare_features(
        val_data.drop(columns=['target_encoded']), fit=False
    )
    y_val_f = val_data['target_encoded'].values

    trainer_f = Trainer(config)
    trainer_f.feature_engineer = full_fe

    tuning_config = config.get('hyperparameter_tuning', {})
    tuning_enabled = tuning_config.get('enabled', False)

    if tuning_enabled:
        trainer_f.train_with_tuning(
            model_type, X_train_f, y_train_f, X_val_f, y_val_f,
            class_names=class_names
        )
    else:
        trainer_f.train_model(
            model_type, X_train_f, y_train_f, X_val_f, y_val_f,
            class_names=class_names
        )

    model_full = trainer_f.trained_models[model_type]
    full_dir = Path('models') / f'saved_future_full_{model_type}'
    trainer_f.save_models(str(full_dir))
    print(f"✓ Full model saved to {full_dir}")

    # --- Future (forecastable) model ---
    print(f"\n--- Training FUTURE (forecastable) model ({model_type}) ---")
    future_fe = FeatureEngineer(config, feature_set='forecastable', climatology=train_climatology)   # pass climatology
    X_train_ff = future_fe.prepare_features(
        train_data.drop(columns=['target_encoded']), fit=True
    )
    y_train_ff = train_data['target_encoded'].values

    X_val_ff = future_fe.prepare_features(
        val_data.drop(columns=['target_encoded']), fit=False
    )
    y_val_ff = val_data['target_encoded'].values

    trainer_ff = Trainer(config)
    trainer_ff.feature_engineer = future_fe

    if tuning_enabled:
        trainer_ff.train_with_tuning(
            model_type, X_train_ff, y_train_ff, X_val_ff, y_val_ff,
            class_names=class_names
        )
    else:
        trainer_ff.train_model(
            model_type, X_train_ff, y_train_ff, X_val_ff, y_val_ff,
            class_names=class_names
        )

    model_future = trainer_ff.trained_models[model_type]
    future_dir = Path('models') / f'saved_future_{model_type}'
    trainer_ff.save_models(str(future_dir))
    print(f"✓ Future model saved to {future_dir}")

    # Return everything needed
    return {
        'full_model': model_full,
        'future_model': model_future,
        'full_fe': full_fe,
        'future_fe': future_fe,
        'trainer_f': trainer_f,
        'trainer_ff': trainer_ff,
        'data_full': (X_train_f, y_train_f, X_val_f, y_val_f),
        'data_future': (X_train_ff, y_train_ff, X_val_ff, y_val_ff)
    }


def load_models(model_type: str, config):
    """Load both model variants and their feature engineers from disk."""
    full_dir = Path('models') / f'saved_future_full_{model_type}'
    future_dir = Path('models') / f'saved_future_{model_type}'

    # Load full model
    model_full = ModelFactory.create_model(model_type, config)
    model_full.load_model(str(full_dir / f'{model_type}.pkl'))
    full_fe = FeatureEngineer(config, feature_set='full')
    full_fe.load(str(full_dir / 'feature_engineer.pkl'))

    # Load future model
    model_future = ModelFactory.create_model(model_type, config)
    model_future.load_model(str(future_dir / f'{model_type}.pkl'))
    future_fe = FeatureEngineer(config, feature_set='forecastable')
    future_fe.load(str(future_dir / 'feature_engineer.pkl'))

    return {
        'full_model': model_full,
        'future_model': model_future,
        'full_fe': full_fe,
        'future_fe': future_fe
    }


def main():
    print("=" * 70)
    print("FUTURE WEATHER EVENT PREDICTION (Live Forecasts) – Interactive Model Selection")
    print("=" * 70)

    config = ConfigLoader('config_future.yaml')
    np.random.seed(config.get('random_seed', 42))
    print("✓ Configuration loaded")

    # --- 1. Ask user which model to use ---
    available_models = ModelFactory.get_available_models()
    print(f"\nAvailable model types: {available_models}")
    default_model = 'xgboost'
    model_type = default_model

    if not model_type:
        model_type = default_model
    if model_type not in available_models:
        print(f"⚠️  Model '{model_type}' not available. Using {default_model}.")
        model_type = default_model

    # --- 2. Check last training time ---
    last_train = get_last_training_time(model_type)
    retrain = False
    if last_train:
        days_ago = (datetime.now() - last_train).days
        print(f"\nModel '{model_type}' was last trained {days_ago} days ago (on {last_train.strftime('%Y-%m-%d %H:%M')}).")
        if days_ago <= 7:
            answer = "n"
            retrain = answer == 'y'
        else:
            print("Model is older than 7 days. Retraining automatically.")
            retrain = True
    else:
        print(f"\nNo existing models found for '{model_type}'. Starting training.")
        retrain = True

    # --- 3. Load or train models ---
    if retrain:
        # Load raw data
        data_loader = DataLoader(config)
        try:
            data = data_loader.load_data()
            print(f"✓ Data loaded: {len(data)} records  ({data['date'].min()} → {data['date'].max()})")
            data = _rename_collector_columns(data)
        except FileNotFoundError:
            print("\n⚠  Data file not found. Ensure data/processed_data.csv exists.")
            return

        # Use config for date splits
        train_end = config.get('data.train_end')      # e.g., "2024-12-31"
        val_start = config.get('data.val_start')      # e.g., "2025-01-01"
        # Convert to datetime for comparison
        train_cutoff = pd.to_datetime(train_end)
        val_cutoff = pd.to_datetime(val_start)

        train_data_raw = data[data['date'] < val_cutoff].copy()   # all before validation start
        val_data_raw   = data[data['date'] >= val_cutoff].copy()

        # If no validation data after cutoff, fallback to last 20% as before (optional)
        if len(val_data_raw) == 0:
            print(f"⚠️  No data from {val_start} onward; using last 20% as validation.")
            split_idx = int(len(data) * 0.8)
            train_data_raw = data.iloc[:split_idx].copy()
            val_data_raw = data.iloc[split_idx:].copy()

        print(f"  Raw training records: {len(train_data_raw)}")
        print(f"  Raw validation records: {len(val_data_raw)}")

        # Create target for each split
        target_creator = TargetCreator(config)
        train_data_with_target = target_creator.create_target(train_data_raw, fit=True)
        val_data_with_target   = target_creator.create_target(val_data_raw,   fit=False)

        # Global label encoding
        label_encoder = LabelEncoder()
        label_encoder.fit(train_data_with_target['weather_event'])
        class_names = label_encoder.classes_.tolist()
        print(f"  Target classes: {class_names} → encoded as {list(range(len(class_names)))}")

        train_data_with_target['target_encoded'] = label_encoder.transform(train_data_with_target['weather_event'])
        val_data_with_target['target_encoded']   = label_encoder.transform(val_data_with_target['weather_event'])

        train_data = train_data_with_target.drop(columns=['weather_event'])
        val_data   = val_data_with_target.drop(columns=['weather_event'])

        # Build climatology from training data only (no leakage)
        train_climatology = build_climatology_from_data(train_data_raw)   # NEW

        # Train both variants
        trained = train_models(model_type, config, train_data, val_data, label_encoder, class_names, train_climatology)
        full_model = trained['full_model']
        future_model = trained['future_model']
        full_fe = trained['full_fe']
        future_fe = trained['future_fe']
        trainer_f = trained['trainer_f']
        trainer_ff = trained['trainer_ff']
        X_train_f, y_train_f, X_val_f, y_val_f = trained['data_full']
        X_train_ff, y_train_ff, X_val_ff, y_val_ff = trained['data_future']
    else:
        # Load existing models
        loaded = load_models(model_type, config)
        full_model = loaded['full_model']
        future_model = loaded['future_model']
        full_fe = loaded['full_fe']
        future_fe = loaded['future_fe']
        # For plots, we still need the validation data. We'll reload it.
        data_loader = DataLoader(config)
        data = data_loader.load_data()
        data = _rename_collector_columns(data)
        # Use same split logic as above
        val_cutoff = pd.to_datetime(config.get('data.val_start'))
        train_data_raw = data[data['date'] < val_cutoff].copy()
        val_data_raw   = data[data['date'] >= val_cutoff].copy()
        if len(val_data_raw) == 0:
            split_idx = int(len(data) * 0.8)
            train_data_raw = data.iloc[:split_idx].copy()
            val_data_raw = data.iloc[split_idx:].copy()
        target_creator = TargetCreator(config)
        train_data_with_target = target_creator.create_target(train_data_raw, fit=True)
        val_data_with_target   = target_creator.create_target(val_data_raw,   fit=False)
        label_encoder = LabelEncoder()
        label_encoder.fit(train_data_with_target['weather_event'])
        class_names = label_encoder.classes_.tolist()
        train_data_with_target['target_encoded'] = label_encoder.transform(train_data_with_target['weather_event'])
        val_data_with_target['target_encoded']   = label_encoder.transform(val_data_with_target['weather_event'])
        train_data = train_data_with_target.drop(columns=['weather_event'])
        val_data   = val_data_with_target.drop(columns=['weather_event'])
        # Prepare validation sets for plotting
        X_val_f = full_fe.prepare_features(val_data.drop(columns=['target_encoded']), fit=False)
        y_val_f = val_data['target_encoded'].values
        X_val_ff = future_fe.prepare_features(val_data.drop(columns=['target_encoded']), fit=False)
        y_val_ff = val_data['target_encoded'].values
        # Dummy trainers for tuning results (if any)
        trainer_f = Trainer(config)
        trainer_ff = Trainer(config)

    # --- 4. Prepare for future prediction ---
    models_dict = {
        'full_model': full_model,
        'future_model': future_model
    }
    feature_engines_dict = {   # Note: parameter name in FuturePredictor is 'feature_engineers_dict'
        'full_model': full_fe,
        'future_model': future_fe
    }

    # Live forecast via Open-Meteo
    print("\n" + "=" * 70)
    print("FUTURE PREDICTION PHASE  (Live Forecast – ENSEMBLE of both variants)")
    print("=" * 70)

    # We need historical data for context, but for climatology we use only training data (no leakage)
    full_historical = data_loader.load_data()   # full dataset for recent context
    full_historical = _rename_collector_columns(full_historical)

    # Build climatology from training data only
    train_only = full_historical[full_historical['date'] < val_cutoff].copy() if 'val_cutoff' in locals() else full_historical
    api_client = OpenMeteoClient(config, historical_data=train_only)   # pass only training data for climatology

    future_predictor = FuturePredictor(
        config,
        models_dict,
        feature_engines_dict,  # correct parameter name
        api_client,
        historical_data=full_historical,   # full historical for recent context (not for climatology)
        random_state=config.get('random_seed', 42),
    )

    cities = full_historical['city'].unique().tolist()
    print(f"🌍 Cities: {cities}")

    #This is where to change the number of day we need and the number of previous days to get from the previous data
    days = 7
    history_days = 60

    future_predictions = future_predictor.predict_future(
        cities=cities,
        start_date=None,
        days=days,
        history_days=history_days
    )

    # ===== FIX: Convert date format to MM/DD/YYYY =====
    if 'date' in future_predictions.columns:
        future_predictions['date'] = pd.to_datetime(future_predictions['date']).dt.strftime('%m/%d/%Y')
        print(f"✓ Date format converted to MM/DD/YYYY (e.g., {future_predictions['date'].iloc[0]})")
    # =================================================

    #This is where to change the paths u need to change this according to your thing
    today_str = datetime.now().strftime('%Y-%m-%d')
    output_files = [
        "../WholeSale-Price-Model/data/raw/Forcast_Weather/upcoming_7_days.csv",
        "../Market_Price_Prediction/content/amika.csv",
        "../Nutrition/Data/Raw/Weather/upcoming_7_days.csv"
    ]
    for file in output_files:
        future_predictor.save_future_predictions(future_predictions, file)

    risks = future_predictor.identify_future_risks(future_predictions, threshold_prob=0.7)
    if len(risks) > 0:
        risk_file = f'results/risks_live_{model_type}_{today_str}_{days}d.csv'
        risks.to_csv(risk_file, index=False)
        print(f"✓ Risk assessment saved → {risk_file}")

    # --- 5. Generate graphs ---
    print("\n" + "=" * 70)
    print("GENERATING MODEL PERFORMANCE GRAPHS")
    print("=" * 70)

    # Evaluate on validation set (using the future model's features for simplicity)
    comparator_future = ModelComparator(config)
    comparison_future = comparator_future.compare_models(
        {'future_model': future_model}, X_val_ff, y_val_ff
    )
    plot_model_comparison(
        comparison_future,
        save_path=f'results/model_comparison_future_{model_type}.png'
    )

    comparator_full = ModelComparator(config)
    comparison_full = comparator_full.compare_models(
        {'full_model': full_model}, X_val_f, y_val_f
    )
    plot_model_comparison(
        comparison_full,
        save_path=f'results/model_comparison_full_{model_type}.png'
    )

    # Overfitting analysis (if available)
    if hasattr(comparator_future, 'overfitting_analysis') and comparator_future.overfitting_analysis is not None:
        plot_overfitting_analysis(
            comparator_future.overfitting_analysis,
            save_path=f'results/overfitting_analysis_future_{model_type}.png'
        )
    if hasattr(comparator_full, 'overfitting_analysis') and comparator_full.overfitting_analysis is not None:
        plot_overfitting_analysis(
            comparator_full.overfitting_analysis,
            save_path=f'results/overfitting_analysis_full_{model_type}.png'
        )

    # Hyperparameter tuning plots (only if tuning was enabled and results exist)
    tuning_config = config.get('hyperparameter_tuning', {})
    if tuning_config.get('enabled', False):
        print("\n📊 Generating hyperparameter tuning plots...")
        if model_type in trainer_ff.tuning_results:
            search_obj = trainer_ff.tuning_results[model_type]['search']
            plot_cv_scores(
                search_obj.cv_results_,
                save_path=f'results/cv_scores_future_{model_type}.png'
            )
        if model_type in trainer_f.tuning_results:
            search_obj = trainer_f.tuning_results[model_type]['search']
            plot_cv_scores(
                search_obj.cv_results_,
                save_path=f'results/cv_scores_full_{model_type}.png'
            )

    # Learning curves (use fresh models without early stopping)
    print(f"\n📈 Generating learning curve for future model ({model_type})...")
    try:
        future_params = future_model.model.get_params()
        if 'early_stopping_rounds' in future_params:
            future_params.pop('early_stopping_rounds')
        fresh_future = ModelFactory.create_model(model_type, config)
        fresh_future.build_model(**future_params)
        cv_splitter = TimeSeriesCV(n_splits=5, gap=30)
        plot_learning_curve(
            fresh_future.model,
            X_train_ff, y_train_ff,
            cv=cv_splitter,
            save_path=f'results/learning_curve_best_future_{model_type}.png'
        )
    except Exception as e:
        print(f"⚠️ Could not generate learning curve for future model: {e}")

    print(f"\n📈 Generating learning curve for full model ({model_type})...")
    try:
        full_params = full_model.model.get_params()
        if 'early_stopping_rounds' in full_params:
            full_params.pop('early_stopping_rounds')
        fresh_full = ModelFactory.create_model(model_type, config)
        fresh_full.build_model(**full_params)
        plot_learning_curve(
            fresh_full.model,
            X_train_f, y_train_f,
            cv=cv_splitter,
            save_path=f'results/learning_curve_best_full_{model_type}.png'
        )
    except Exception as e:
        print(f"⚠️ Could not generate learning curve for full model: {e}")

    print("\n" + "=" * 70)
    print(f"✅  FUTURE PREDICTION COMPLETE – graphs saved to results/ (model: {model_type})")
    print("=" * 70)


if __name__ == '__main__':
    main()