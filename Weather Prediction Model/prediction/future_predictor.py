"""
Future weather event prediction module.
Uses historical data (from collector) for context + live forecasts.
OUTPUT FORMAT: date, city, prob_drought, prob_flood, prob_normal, predicted_class
"""
import pandas as pd
import numpy as np
from pathlib import Path


class FuturePredictor:
    def __init__(self, config, models_dict, feature_engineers_dict, api_client,
                 historical_data, random_state=None):
        """
        Args:
            config           : ConfigLoader instance.
            models_dict      : Dict {name: trained_model}.
            feature_engineers_dict : Dict {name: fitted FeatureEngineer}.
            api_client       : OpenMeteoClient (for forecasts).
            historical_data  : Full historical DataFrame (used to extract recent context).
            random_state     : Optional seed.
        """
        self.config = config
        self.models = models_dict
        self.feature_engineers = feature_engineers_dict
        self.api_client = api_client
        self.historical_data = historical_data
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)

        # Determine the original class names from the first model
        if models_dict:
            first_model = list(models_dict.values())[0]
            # Prefer original_classes if stored, otherwise fallback to classes_ (which may be ints)
            if hasattr(first_model, 'original_classes') and first_model.original_classes is not None:
                self.class_names = first_model.original_classes
                print(f"Using original class names from model: {self.class_names}")
            elif hasattr(first_model, 'classes_'):
                self.class_names = first_model.classes_.tolist()
                print(f"Using model.classes_ (may be integers): {self.class_names}")
            else:
                self.class_names = ['drought', 'flood_risk', 'normal']
                print(f"Using default class names: {self.class_names}")
        else:
            self.class_names = []

    def _get_recent_history(self, cities, history_days=60):
        """
        Extract the last `history_days` of historical data for each city.
        Returns a DataFrame sorted by city and date.
        """
        df_hist = self.historical_data.copy()
        df_hist['date'] = pd.to_datetime(df_hist['date'])
        df_hist = df_hist[df_hist['city'].isin(cities)]

        recent = []
        for city in cities:
            city_data = df_hist[df_hist['city'] == city].sort_values('date')
            if len(city_data) > history_days:
                city_data = city_data.iloc[-history_days:]
            recent.append(city_data)
        if recent:
            return pd.concat(recent).sort_values(['city', 'date']).reset_index(drop=True)
        else:
            return pd.DataFrame()

    def predict_future(self, cities, start_date=None, days=14, history_days=60):
        """
        Combine recent historical context + forecast, prepare features for each model,
        average probabilities, and return predictions for the forecast period.
        """
        # 1. Recent history
        print(f"\n📜 Using last {history_days} days of historical data as context...")
        hist_context = self._get_recent_history(cities, history_days=history_days)
        if hist_context.empty:
            print("⚠️  No historical context available. Predictions may be less accurate.")

        # 2. Future forecast from API
        print(f"🌤️ Fetching {days}-day forecast from Open-Meteo...")
        forecast = self.api_client.get_forecast(cities, start_date=start_date, days=days)

        # 3. Combine
        if not hist_context.empty:
            common_cols = list(set(hist_context.columns) & set(forecast.columns))
            combined = pd.concat([hist_context[common_cols], forecast[common_cols]], ignore_index=True)
            combined = combined.sort_values(['city', 'date']).reset_index(drop=True)
        else:
            combined = forecast.copy()

        # 4. Add anomaly features
        if self.api_client.climatology is not None:
            combined = self.api_client.add_anomaly_features(combined)

        # 5. Prepare features and get probabilities
        all_probs = []
        for model_name, model in self.models.items():
            print(f"\n📊 Preparing features for model: {model_name}")
            feat_eng = self.feature_engineers[model_name]
            X_all = feat_eng.prepare_features(combined.copy(), fit=False)
            probs_all = model.predict_proba(X_all)
            all_probs.append(probs_all)

        avg_probs = np.mean(all_probs, axis=0)
        class_names = self.class_names

        # 6. Identify forecast rows
        today = pd.Timestamp.now().normalize()
        forecast_mask = combined['date'] >= today

        results = combined[forecast_mask][['date', 'city']].copy()
        for i, class_name in enumerate(class_names):
            # Use the original class name for the probability column
            col_name = f'prob_{class_name}'
            results[col_name] = avg_probs[forecast_mask, i]

        # Determine predicted class (index of max probability)
        pred_indices = np.argmax(avg_probs[forecast_mask, :], axis=1)
        results['predicted_class_raw'] = [class_names[i] for i in pred_indices]

        # Sanity check: if any probability row sums to zero, warn
        prob_sum = results[[f'prob_{c}' for c in class_names]].sum(axis=1)
        if (prob_sum == 0).any():
            print("⚠️  Warning: Some rows have zero total probability. Check feature engineering or model predictions.")

        print(f"\n✅ Ensemble predictions generated for {len(results)} future days.")
        return results

    @staticmethod
    def save_future_predictions(predictions, output_file='results/predictions_ensemble.csv'):
        """
        Save predictions to CSV with the required format:
        date, city, prob_drought, prob_flood, prob_normal, predicted_class
        """
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        df = predictions.copy()

        # Map internal class names to output names
        # If class names are ['drought','flood_risk','normal'], rename prob_flood_risk -> prob_flood
        # and predicted_class_raw 'flood_risk' -> 'flood'
        if 'prob_flood_risk' in df.columns:
            df.rename(columns={'prob_flood_risk': 'prob_flood'}, inplace=True)

        if 'predicted_class_raw' in df.columns:
            df['predicted_class'] = df['predicted_class_raw'].replace('flood_risk', 'flood')
            df.drop(columns=['predicted_class_raw'], inplace=True)
        else:
            # Fallback: if predicted_class_raw missing, assume predicted_class is already correct
            if 'predicted_class' not in df.columns:
                raise ValueError("No predicted_class column found. Cannot save predictions.")

        # Ensure all required probability columns exist
        required_probs = ['prob_drought', 'prob_flood', 'prob_normal']
        for col in required_probs:
            if col not in df.columns:
                print(f"⚠️  Warning: {col} missing, adding with zeros.")
                df[col] = 0.0

        # Round probabilities
        for col in required_probs:
            df[col] = df[col].round(3)

        # Reorder columns
        final_cols = ['date', 'city'] + required_probs + ['predicted_class']
        df = df[final_cols]

        # Ensure predicted_class is string
        df['predicted_class'] = df['predicted_class'].astype(str)

        df.to_csv(output_file, index=False)
        print(f"\n✅ Future predictions saved to: {output_file}")
        print("\nPredicted class distribution:")
        print(df['predicted_class'].value_counts())
        return output_file

    def identify_future_risks(self, predictions, threshold_prob=0.7):
        """
        Identify high-risk periods in future predictions.
        """
        risks = []
        class_map = {'flood_risk': 'flood', 'drought': 'drought', 'normal': 'normal'}

        for city in predictions['city'].unique():
            city_data = predictions[predictions['city'] == city].sort_values('date')

            for class_name in self.class_names:
                if class_name == 'normal':
                    continue

                prob_col = f'prob_{class_name}'
                if prob_col not in city_data.columns:
                    continue

                risk_mask = (city_data['predicted_class_raw'] == class_name) & (city_data[prob_col] >= threshold_prob)
                risk_days = city_data[risk_mask]

                if len(risk_days) > 0:
                    risks.append({
                        'city': city,
                        'risk_type': class_map.get(class_name, class_name),
                        'n_days': len(risk_days),
                        'avg_probability': risk_days[prob_col].mean(),
                        'first_date': risk_days['date'].min(),
                        'last_date': risk_days['date'].max()
                    })

        risks_df = pd.DataFrame(risks)

        if len(risks_df) > 0:
            print("\n⚠️  HIGH RISK PERIODS IDENTIFIED:")
            print(risks_df.to_string(index=False))
        else:
            print("\n✅ No high-risk periods identified")

        return risks_df