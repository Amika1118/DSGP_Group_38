"""
Prediction module for generating weather event predictions.
"""
import pandas as pd
import numpy as np
from pathlib import Path


class Predictor:
    """Make predictions and generate reports for teammates."""

    def __init__(self, config):
        """Initialize predictor."""
        self.config = config
        self.predictions = None
        self.extreme_events = None

    def predict(self, model, X, data):
        """Make predictions using trained model."""
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)

        results = data[['date', 'city']].copy()
        results['predicted_class'] = y_pred

        for i, class_name in enumerate(model.classes_):
            results[f'prob_{class_name}'] = y_proba[:, i]

        self.predictions = results
        return results

    def save_predictions_for_teammate(self, output_file=None):
        """Save predictions in teammate-friendly CSV format."""
        if self.predictions is None:
            raise ValueError("No predictions available")

        if output_file is None:
            output_file = 'predictions_for_teammate.csv'

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        class_cols = [col for col in self.predictions.columns if col.startswith('prob_')]
        column_order = ['date', 'city'] + class_cols + ['predicted_class']
        output_df = self.predictions[column_order].copy()

        for col in class_cols:
            output_df[col] = output_df[col].round(2)

        output_df.to_csv(output_file, index=False)
        print(f"\nPredictions saved to: {output_file}")
        print(f"Total predictions: {len(output_df)}")
        return output_file

    def identify_extreme_events(self, predictions=None, threshold_prob=0.7):
        """
        Identify extreme weather events from predictions.

        Args:
            predictions: DataFrame with predictions (optional, uses self.predictions if None)
            threshold_prob: Minimum probability threshold for extreme events

        Returns:
            DataFrame with extreme events
        """
        # Use provided predictions or fall back to stored predictions
        if predictions is not None:
            pred_data = predictions
        elif self.predictions is not None:
            pred_data = self.predictions
        else:
            raise ValueError("No predictions available")

        events = []

        for city in pred_data['city'].unique():
            city_data = pred_data[pred_data['city'] == city].sort_values('date')

            # Drought events
            if 'prob_drought' in city_data.columns:
                drought_mask = ((city_data['predicted_class'] == 'drought') &
                              (city_data['prob_drought'] >= threshold_prob))
                if drought_mask.any():
                    events.extend(self._extract_events(city_data[drought_mask], 'drought', city))

            # Flood events
            if 'prob_flood_risk' in city_data.columns:
                flood_mask = ((city_data['predicted_class'] == 'flood_risk') &
                            (city_data['prob_flood_risk'] >= threshold_prob))
                if flood_mask.any():
                    events.extend(self._extract_events(city_data[flood_mask], 'flood_risk', city))

        extreme_events = pd.DataFrame(events)

        # Store in self.extreme_events
        self.extreme_events = extreme_events

        return extreme_events

    def _extract_events(self, data, event_type, city):
        """Extract continuous event periods."""
        events = []
        if len(data) == 0:
            return events

        data = data.sort_values('date').reset_index(drop=True)
        start_date = data.iloc[0]['date']
        prev_date = start_date
        max_prob = data.iloc[0][f'prob_{event_type}']

        for i in range(1, len(data)):
            current_date = data.iloc[i]['date']
            current_prob = data.iloc[i][f'prob_{event_type}']

            if (pd.to_datetime(current_date) - pd.to_datetime(prev_date)).days > 2:
                events.append({
                    'city': city,
                    'event_type': event_type,
                    'start_date': start_date,
                    'end_date': prev_date,
                    'duration_days': (pd.to_datetime(prev_date) - pd.to_datetime(start_date)).days + 1,
                    'max_probability': round(max_prob, 3)
                })
                start_date = current_date
                max_prob = current_prob
            else:
                max_prob = max(max_prob, current_prob)

            prev_date = current_date

        events.append({
            'city': city,
            'event_type': event_type,
            'start_date': start_date,
            'end_date': prev_date,
            'duration_days': (pd.to_datetime(prev_date) - pd.to_datetime(start_date)).days + 1,
            'max_probability': round(max_prob, 3)
        })

        return events

    def save_extreme_events_report(self, output_file='extreme_events_report.csv'):
        """Save extreme events report."""
        if self.extreme_events is None:
            # Identify extreme events first if not already done
            self.extreme_events = self.identify_extreme_events()

        # Create output directory
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        # Save to CSV
        self.extreme_events.to_csv(output_file, index=False)
        print(f"\nExtreme events report saved to: {output_file}")
        print(f"Total events: {len(self.extreme_events)}")
        return output_file


if __name__ == '__main__':
    print("Predictor module created successfully")