# World-Class Model Evaluation Framework

## Mission-Critical Directive

> **"Make absolute world class models - the future of my company depends on it!"**

This framework ensures **every model** goes through rigorous evaluation, iteration, and improvement cycles to achieve world-class performance.

---

## 1. Evaluation Philosophy

### Three-Stage Evaluation Process

```
Stage 1: TRAIN → Baseline performance on validation set
   ↓
Stage 2: EVALUATE → Comprehensive metrics + failure analysis
   ↓
Stage 3: ITERATE → Improve model based on insights
   ↓
Repeat until WORLD-CLASS threshold achieved
```

### World-Class Standards

| Model | Metric | Industry Benchmark | Our Target | World-Class |
|-------|--------|-------------------|------------|-------------|
| **Model 1: DA Price** | MAE | $5-8/MWh | < $5/MWh | < $3/MWh |
| | R² | 0.85-0.90 | > 0.85 | > 0.92 |
| **Model 2: RT Price** | MAE | $15-25/MWh | < $15/MWh | < $10/MWh |
| | R² | 0.75-0.85 | > 0.75 | > 0.88 |
| **Model 3: Price Spike** | AUC | 0.88 | > 0.88 | > 0.92 |
| | Precision@5% | 60-80% | > 60% | > 85% |
| | Recall@90% | 85-90% | > 90% | > 95% |
| **Models 4-7: AS Prices** | MAE | Varies | TBD | Top 10% |

---

## 2. Comprehensive Evaluation Metrics

### 2.1 Model 3: RT Price Spike Prediction (Most Critical)

#### Primary Metrics
```python
class SpikeModelEvaluator:
    """Comprehensive evaluation for price spike prediction."""

    def evaluate(self, model, test_df):
        predictions = model.predict(test_df)
        true_labels = test_df['price_spike']

        metrics = {
            # 1. PRIMARY: AUC-ROC
            'auc_roc': roc_auc_score(true_labels, predictions),

            # 2. Precision-Recall Curve
            'auc_pr': average_precision_score(true_labels, predictions),

            # 3. Precision at top 5% predictions
            'precision_at_5pct': self.precision_at_k(predictions, true_labels, k=0.05),

            # 4. Recall at 90% precision threshold
            'recall_at_90_precision': self.recall_at_precision(predictions, true_labels, 0.90),

            # 5. F1 Score (optimal threshold)
            'f1_score': self.best_f1_score(predictions, true_labels),

            # 6. Calibration (predicted prob vs. actual freq)
            'calibration_error': self.calibration_error(predictions, true_labels),

            # 7. Financial metrics
            'financial_value': self.financial_impact(predictions, true_labels, test_df)
        }

        return metrics
```

#### Financial Impact Evaluation
```python
def financial_impact(self, predictions, true_labels, test_df):
    """
    Measure model value in dollars:
    - True Positive: Predict spike → Actually spikes → PROFIT
    - False Positive: Predict spike → No spike → LOSS (missed opportunity)
    - False Negative: Miss spike → HUGE LOSS (no positioning)
    - True Negative: Predict no spike → No spike → OK
    """

    # Assume trading strategy: If spike_prob > 0.7, hold/charge battery
    threshold = 0.7
    predicted_spikes = predictions > threshold

    true_spikes = true_labels == 1
    prices = test_df['rt_lmp']

    # True Positives: Correctly predicted spikes
    tp_mask = predicted_spikes & true_spikes
    tp_value = (prices[tp_mask] * 10).sum()  # 10 MW battery, discharge at spike

    # False Negatives: Missed spikes (HUGE COST)
    fn_mask = ~predicted_spikes & true_spikes
    fn_cost = (prices[fn_mask] * 10).sum()  # Missed revenue

    # False Positives: Predicted spike but didn't happen
    fp_mask = predicted_spikes & ~true_spikes
    fp_cost = (prices[fp_mask] * 2).sum()  # Opportunity cost of waiting

    total_value = tp_value - fn_cost - fp_cost

    return {
        'total_value': total_value,
        'tp_value': tp_value,
        'fn_cost': fn_cost,
        'fp_cost': fp_cost,
        'value_per_day': total_value / (len(test_df) / 288),  # 5-min intervals
    }
```

#### Failure Analysis
```python
def analyze_failures(self, model, test_df):
    """Deep dive into model failures."""

    predictions = model.predict(test_df)
    true_labels = test_df['price_spike']

    # Find False Negatives (missed spikes)
    fn_mask = (predictions < 0.5) & (true_labels == 1)
    false_negatives = test_df[fn_mask]

    # Find False Positives (false alarms)
    fp_mask = (predictions > 0.5) & (true_labels == 0)
    false_positives = test_df[fp_mask]

    # Analyze FN patterns
    fn_analysis = {
        'count': len(false_negatives),
        'avg_price': false_negatives['rt_lmp'].mean(),
        'max_price': false_negatives['rt_lmp'].max(),

        # Feature analysis: Why did we miss?
        'avg_load_error': false_negatives['load_error_mw'].mean(),
        'avg_wind_error': false_negatives['wind_error_mw'].mean(),
        'avg_temp': false_negatives['temp'].mean(),
        'hours': false_negatives['hour'].value_counts().to_dict(),

        # Most costly misses
        'top_10_costly_misses': false_negatives.nlargest(10, 'rt_lmp')[
            ['timestamp', 'rt_lmp', 'load_error_mw', 'wind_error_mw', 'temp']
        ].to_dict('records')
    }

    # Analyze FP patterns
    fp_analysis = {
        'count': len(false_positives),
        'avg_predicted_prob': predictions[fp_mask].mean(),

        # Why did we predict spike when there wasn't?
        'avg_load_error': false_positives['load_error_mw'].mean(),
        'avg_wind_error': false_positives['wind_error_mw'].mean(),

        # Most confident wrong predictions
        'top_10_confident_errors': false_positives.nlargest(10, predictions[fp_mask])[
            ['timestamp', 'rt_lmp', 'load_error_mw', 'wind_error_mw']
        ].to_dict('records')
    }

    return {
        'false_negatives': fn_analysis,
        'false_positives': fp_analysis,
        'recommendations': self.generate_improvement_recommendations(fn_analysis, fp_analysis)
    }
```

#### Temporal Analysis
```python
def temporal_performance(self, model, test_df):
    """Evaluate performance across time periods."""

    predictions = model.predict(test_df)
    true_labels = test_df['price_spike']

    # By month
    monthly_auc = test_df.groupby(test_df['timestamp'].dt.month).apply(
        lambda x: roc_auc_score(x['price_spike'], predictions[x.index])
    )

    # By hour
    hourly_auc = test_df.groupby(test_df['timestamp'].dt.hour).apply(
        lambda x: roc_auc_score(x['price_spike'], predictions[x.index])
    )

    # By season
    test_df['season'] = test_df['timestamp'].dt.month.map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    })
    seasonal_auc = test_df.groupby('season').apply(
        lambda x: roc_auc_score(x['price_spike'], predictions[x.index])
    )

    # During extreme weather
    heatwave_auc = roc_auc_score(
        test_df[test_df['heat_wave'] == 1]['price_spike'],
        predictions[test_df['heat_wave'] == 1]
    )

    return {
        'monthly': monthly_auc.to_dict(),
        'hourly': hourly_auc.to_dict(),
        'seasonal': seasonal_auc.to_dict(),
        'extreme_weather': {
            'heatwave_auc': heatwave_auc,
            'total_heatwave_hours': (test_df['heat_wave'] == 1).sum()
        },
        'worst_performing_periods': self.identify_weak_periods(monthly_auc)
    }
```

---

### 2.2 Model 1 & 2: Price Forecasting

#### Regression Metrics
```python
class PriceModelEvaluator:
    """Comprehensive evaluation for price forecasting models."""

    def evaluate(self, model, test_df, model_type='da'):
        predictions = model.predict(test_df)
        true_prices = test_df['da_lmp'] if model_type == 'da' else test_df['rt_lmp']

        metrics = {
            # 1. Mean Absolute Error (PRIMARY)
            'mae': mean_absolute_error(true_prices, predictions),

            # 2. Root Mean Squared Error
            'rmse': np.sqrt(mean_squared_error(true_prices, predictions)),

            # 3. R² Score
            'r2': r2_score(true_prices, predictions),

            # 4. Mean Absolute Percentage Error
            'mape': np.mean(np.abs((true_prices - predictions) / true_prices)) * 100,

            # 5. Quantile metrics (capture distribution)
            'quantile_losses': self.quantile_loss(predictions, true_prices),

            # 6. Peak hour performance (most valuable)
            'peak_mae': self.peak_hour_mae(predictions, true_prices, test_df),

            # 7. Extreme price performance
            'high_price_mae': self.extreme_price_mae(predictions, true_prices, threshold=500),

            # 8. Directional accuracy (up/down)
            'directional_accuracy': self.directional_accuracy(predictions, true_prices),

            # 9. Financial metrics
            'trading_pnl': self.trading_simulation(predictions, true_prices, test_df)
        }

        return metrics
```

#### Peak Hour Performance
```python
def peak_hour_mae(self, predictions, true_prices, test_df):
    """
    Peak hours (4 PM - 8 PM) are most valuable.
    Model MUST perform well here.
    """

    peak_hours = [16, 17, 18, 19, 20]
    peak_mask = test_df['timestamp'].dt.hour.isin(peak_hours)

    peak_mae = mean_absolute_error(
        true_prices[peak_mask],
        predictions[peak_mask]
    )

    # Compare to off-peak
    offpeak_mae = mean_absolute_error(
        true_prices[~peak_mask],
        predictions[~peak_mask]
    )

    return {
        'peak_mae': peak_mae,
        'offpeak_mae': offpeak_mae,
        'peak_vs_offpeak_ratio': peak_mae / offpeak_mae
    }
```

#### Trading Simulation
```python
def trading_simulation(self, predictions, true_prices, test_df):
    """
    Simulate battery trading decisions based on forecasts.

    Strategy:
    - If forecast_price > threshold, discharge
    - If forecast_price < threshold, charge
    - Compare PnL vs. perfect foresight
    """

    # Simple strategy: Discharge when forecast > $75/MWh
    discharge_threshold = 75
    charge_threshold = 30

    discharge_decisions = predictions > discharge_threshold
    charge_decisions = predictions < charge_threshold

    # Calculate PnL
    discharge_pnl = (true_prices[discharge_decisions]).sum() * 10  # 10 MW battery
    charge_pnl = -(true_prices[charge_decisions]).sum() * 10  # Cost of charging

    total_pnl = discharge_pnl + charge_pnl

    # Perfect foresight PnL (upper bound)
    perfect_discharge = true_prices > discharge_threshold
    perfect_charge = true_prices < charge_threshold

    perfect_pnl = (
        (true_prices[perfect_discharge]).sum() * 10 -
        (true_prices[perfect_charge]).sum() * 10
    )

    return {
        'forecast_pnl': total_pnl,
        'perfect_pnl': perfect_pnl,
        'pnl_ratio': total_pnl / perfect_pnl if perfect_pnl > 0 else 0,
        'discharge_accuracy': (discharge_decisions == perfect_discharge).mean(),
        'num_trades': discharge_decisions.sum() + charge_decisions.sum()
    }
```

---

### 2.3 Cross-Model Validation

```python
def cross_model_consistency(spike_model, da_model, rt_model, test_df):
    """
    Check if models are consistent with each other.

    Example:
    - If spike model predicts high probability → RT model should predict high price
    - If DA model predicts high price → Spike model should predict higher probability
    """

    spike_probs = spike_model.predict(test_df)
    da_predictions = da_model.predict(test_df)
    rt_predictions = rt_model.predict(test_df)

    # Correlation analysis
    corr_spike_rt = np.corrcoef(spike_probs, rt_predictions)[0, 1]
    corr_spike_da = np.corrcoef(spike_probs, da_predictions)[0, 1]
    corr_da_rt = np.corrcoef(da_predictions, rt_predictions)[0, 1]

    # Consistency checks
    high_spike_prob = spike_probs > 0.7
    avg_rt_price_high_spike = rt_predictions[high_spike_prob].mean()
    avg_rt_price_low_spike = rt_predictions[~high_spike_prob].mean()

    return {
        'correlations': {
            'spike_vs_rt': corr_spike_rt,
            'spike_vs_da': corr_spike_da,
            'da_vs_rt': corr_da_rt
        },
        'consistency': {
            'rt_price_when_spike_high': avg_rt_price_high_spike,
            'rt_price_when_spike_low': avg_rt_price_low_spike,
            'ratio': avg_rt_price_high_spike / avg_rt_price_low_spike
        },
        'warnings': self.generate_consistency_warnings(corr_spike_rt, corr_spike_da)
    }
```

---

## 3. Iteration & Improvement Strategies

### 3.1 Hyperparameter Optimization

```python
import optuna

def optimize_model_3(train_df, val_df):
    """Use Optuna for Bayesian hyperparameter optimization."""

    def objective(trial):
        # Hyperparameters to tune
        d_model = trial.suggest_categorical('d_model', [256, 512, 768])
        nhead = trial.suggest_categorical('nhead', [4, 8, 16])
        num_layers = trial.suggest_int('num_layers', 4, 8)
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
        batch_size = trial.suggest_categorical('batch_size', [128, 256, 512])

        # Focal loss parameters
        focal_alpha = trial.suggest_float('focal_alpha', 0.5, 0.9)
        focal_gamma = trial.suggest_float('focal_gamma', 1.0, 3.0)

        # Train model
        model = PriceSpikeTransformer(
            input_dim=len(feature_cols),
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout
        )

        trainer = PriceSpikeModelTrainer(device='cuda')
        model, history = trainer.train(
            train_df, val_df,
            epochs=50,  # Shorter for HPO
            batch_size=batch_size,
            lr=lr,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma
        )

        # Optimize for best validation AUC
        best_auc = max(history['val_auc'])

        return best_auc

    # Run optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100, timeout=86400)  # 24 hours

    print(f"Best AUC: {study.best_value}")
    print(f"Best params: {study.best_params}")

    return study.best_params
```

### 3.2 Feature Engineering Iteration

```python
def feature_importance_analysis(model, train_df, feature_cols):
    """
    Identify which features matter most.

    Use:
    1. SHAP values (model-agnostic)
    2. Permutation importance
    3. Ablation studies (remove feature, measure drop)
    """

    import shap

    # SHAP values
    explainer = shap.DeepExplainer(model, train_df[feature_cols].values[:1000])
    shap_values = explainer.shap_values(train_df[feature_cols].values[1000:2000])

    # Average absolute SHAP values
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('importance', ascending=False)

    # Recommendations
    print("\n🔍 FEATURE IMPORTANCE ANALYSIS\n")
    print("Top 10 Most Important Features:")
    print(feature_importance.head(10))

    print("\n❌ Features to Consider Removing (low importance):")
    print(feature_importance.tail(10))

    # Ablation study
    baseline_auc = evaluate_model(model, val_df)['auc_roc']

    ablation_results = []
    for feature in feature_importance.head(10)['feature']:
        # Remove feature and retrain
        reduced_features = [f for f in feature_cols if f != feature]
        reduced_model = train_model_with_features(train_df, val_df, reduced_features)
        reduced_auc = evaluate_model(reduced_model, val_df)['auc_roc']

        ablation_results.append({
            'feature': feature,
            'auc_drop': baseline_auc - reduced_auc
        })

    ablation_df = pd.DataFrame(ablation_results).sort_values('auc_drop', ascending=False)

    print("\n📉 ABLATION STUDY (AUC drop when feature removed):")
    print(ablation_df)

    return feature_importance, ablation_df
```

### 3.3 Ensemble Methods

```python
def create_ensemble(models_list, val_df):
    """
    Combine multiple models for better performance.

    Strategies:
    1. Simple averaging
    2. Weighted averaging (based on validation AUC)
    3. Stacking (meta-model learns to combine)
    """

    # Strategy 1: Simple average
    ensemble_preds_avg = np.mean([m.predict(val_df) for m in models_list], axis=0)
    auc_avg = roc_auc_score(val_df['price_spike'], ensemble_preds_avg)

    # Strategy 2: Weighted by validation AUC
    weights = np.array([evaluate_model(m, val_df)['auc_roc'] for m in models_list])
    weights = weights / weights.sum()

    ensemble_preds_weighted = np.average(
        [m.predict(val_df) for m in models_list],
        axis=0,
        weights=weights
    )
    auc_weighted = roc_auc_score(val_df['price_spike'], ensemble_preds_weighted)

    # Strategy 3: Stacking
    meta_features = np.column_stack([m.predict(val_df) for m in models_list])
    meta_model = LogisticRegression()
    meta_model.fit(meta_features, val_df['price_spike'])
    ensemble_preds_stacked = meta_model.predict_proba(meta_features)[:, 1]
    auc_stacked = roc_auc_score(val_df['price_spike'], ensemble_preds_stacked)

    print(f"\n🎯 ENSEMBLE RESULTS:")
    print(f"  Simple Average:   AUC = {auc_avg:.4f}")
    print(f"  Weighted Average: AUC = {auc_weighted:.4f}")
    print(f"  Stacked:          AUC = {auc_stacked:.4f}")

    # Return best ensemble
    best_method = max([
        ('average', auc_avg, ensemble_preds_avg),
        ('weighted', auc_weighted, ensemble_preds_weighted),
        ('stacked', auc_stacked, ensemble_preds_stacked)
    ], key=lambda x: x[1])

    print(f"\n✅ Best Ensemble: {best_method[0]} with AUC = {best_method[1]:.4f}")

    return best_method
```

### 3.4 Data Augmentation

```python
def augment_spike_data(train_df):
    """
    Address class imbalance by augmenting spike examples.

    Techniques:
    1. SMOTE (Synthetic Minority Oversampling)
    2. Time series augmentation (add noise, scale, shift)
    3. Mixup (interpolate between examples)
    """

    from imblearn.over_sampling import SMOTE

    # SMOTE
    smote = SMOTE(sampling_strategy=0.1, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(
        train_df[feature_cols],
        train_df['price_spike']
    )

    # Time series augmentation for spike examples
    spike_examples = train_df[train_df['price_spike'] == 1]

    augmented_spikes = []
    for _, row in spike_examples.iterrows():
        # Add Gaussian noise
        noisy_row = row.copy()
        noisy_row[numerical_features] += np.random.normal(0, 0.05, len(numerical_features))
        augmented_spikes.append(noisy_row)

        # Scale features (±10%)
        scaled_row = row.copy()
        scaled_row[numerical_features] *= np.random.uniform(0.9, 1.1)
        augmented_spikes.append(scaled_row)

    augmented_df = pd.concat([train_df, pd.DataFrame(augmented_spikes)], ignore_index=True)

    print(f"Original spike examples: {len(spike_examples)}")
    print(f"Augmented spike examples: {len(augmented_spikes)}")
    print(f"New training set size: {len(augmented_df)}")
    print(f"New spike rate: {augmented_df['price_spike'].mean()*100:.2f}%")

    return augmented_df
```

---

## 4. Automated Improvement Loop

```python
class WorldClassModelTrainer:
    """
    Automated training loop with evaluation and iteration.

    Runs until model reaches world-class performance.
    """

    def __init__(self, target_metric='auc', target_value=0.92):
        self.target_metric = target_metric
        self.target_value = target_value
        self.iteration = 0
        self.best_model = None
        self.best_score = 0

    def train_until_world_class(self, train_df, val_df, test_df):
        """
        Main training loop with automatic improvement.
        """

        print(f"\n{'='*80}")
        print(f"🎯 TARGET: {self.target_metric.upper()} > {self.target_value}")
        print(f"{'='*80}\n")

        while self.best_score < self.target_value and self.iteration < 10:
            self.iteration += 1

            print(f"\n{'='*80}")
            print(f"🔄 ITERATION {self.iteration}")
            print(f"{'='*80}\n")

            # Step 1: Train model
            model, history = self.train_model(train_df, val_df)

            # Step 2: Comprehensive evaluation
            eval_results = self.evaluate_model(model, val_df, test_df)

            current_score = eval_results[self.target_metric]
            print(f"\n📊 Current {self.target_metric.upper()}: {current_score:.4f}")
            print(f"   Target: {self.target_value:.4f}")
            print(f"   Gap: {self.target_value - current_score:.4f}")

            # Step 3: Update best model
            if current_score > self.best_score:
                self.best_score = current_score
                self.best_model = model
                print(f"   ✅ NEW BEST MODEL!")

            # Step 4: Failure analysis
            failures = self.analyze_failures(model, val_df)

            # Step 5: Generate improvement recommendations
            recommendations = self.generate_recommendations(eval_results, failures)

            # Step 6: Apply improvements
            train_df, val_df = self.apply_improvements(train_df, val_df, recommendations)

            # Step 7: Hyperparameter tuning
            if self.iteration % 3 == 0:
                best_params = self.run_hyperparameter_optimization(train_df, val_df)
                self.update_model_config(best_params)

            # Check if world-class achieved
            if self.best_score >= self.target_value:
                print(f"\n{'='*80}")
                print(f"🏆 WORLD-CLASS MODEL ACHIEVED!")
                print(f"   {self.target_metric.upper()}: {self.best_score:.4f} (Target: {self.target_value:.4f})")
                print(f"   Iterations: {self.iteration}")
                print(f"{'='*80}\n")
                break

        # Final evaluation on test set
        final_results = self.evaluate_model(self.best_model, test_df, test_df)

        print(f"\n{'='*80}")
        print(f"🎉 FINAL TEST SET RESULTS")
        print(f"{'='*80}")
        for metric, value in final_results.items():
            print(f"  {metric}: {value:.4f}")
        print(f"{'='*80}\n")

        return self.best_model, final_results

    def generate_recommendations(self, eval_results, failures):
        """
        Automatically generate improvement recommendations based on analysis.
        """

        recommendations = []

        # Check feature importance
        if eval_results['feature_importance_analyzed']:
            low_importance_features = eval_results['low_importance_features']
            if len(low_importance_features) > 5:
                recommendations.append({
                    'type': 'remove_features',
                    'features': low_importance_features,
                    'reason': 'Low importance features adding noise'
                })

        # Check failure patterns
        if failures['false_negative_rate'] > 0.15:
            recommendations.append({
                'type': 'augment_spike_data',
                'target': 'increase spike examples',
                'reason': f'High false negative rate: {failures["false_negative_rate"]:.2%}'
            })

        # Check temporal performance
        if eval_results['worst_month_auc'] < 0.80:
            recommendations.append({
                'type': 'add_seasonal_features',
                'reason': f'Poor performance in {eval_results["worst_month"]}'
            })

        # Check calibration
        if eval_results['calibration_error'] > 0.10:
            recommendations.append({
                'type': 'calibration',
                'method': 'isotonic_regression',
                'reason': f'Poor calibration: {eval_results["calibration_error"]:.3f}'
            })

        return recommendations

    def apply_improvements(self, train_df, val_df, recommendations):
        """
        Automatically apply recommended improvements.
        """

        for rec in recommendations:
            print(f"\n🔧 Applying: {rec['type']}")
            print(f"   Reason: {rec['reason']}")

            if rec['type'] == 'remove_features':
                # Remove low-importance features
                features_to_remove = rec['features']
                train_df = train_df.drop(columns=features_to_remove)
                val_df = val_df.drop(columns=features_to_remove)
                print(f"   Removed {len(features_to_remove)} features")

            elif rec['type'] == 'augment_spike_data':
                # Augment spike examples
                train_df = augment_spike_data(train_df)
                print(f"   New spike rate: {train_df['price_spike'].mean()*100:.2f}%")

            elif rec['type'] == 'add_seasonal_features':
                # Add more temporal features
                train_df = add_seasonal_features(train_df)
                val_df = add_seasonal_features(val_df)
                print(f"   Added seasonal interaction features")

            elif rec['type'] == 'calibration':
                # Apply calibration to model predictions
                self.use_calibration = True
                print(f"   Will apply {rec['method']} calibration")

        return train_df, val_df
```

---

## 5. Testing Protocols

### 5.1 Backtesting on Historical Events

```python
def backtest_historical_events(model):
    """
    Test model on specific historical events.

    Events to test (EXCLUDE Winter Storm Uri):
    - Summer 2023 heat waves (June-August)
    - Summer 2024 heat waves
    - Spring/Fall shoulder months (low load, high renewables)
    - Extreme wind/solar forecast errors
    """

    events = {
        'Summer_2023_Heatwave': {
            'start': '2023-06-15',
            'end': '2023-08-31',
            'expected_spikes': 50,
            'min_auc': 0.85
        },
        'Summer_2024_Heatwave': {
            'start': '2024-06-15',
            'end': '2024-08-31',
            'expected_spikes': 45,
            'min_auc': 0.85
        },
        'Spring_2024_Renewables_High': {
            'start': '2024-03-01',
            'end': '2024-05-31',
            'expected_spikes': 10,
            'min_auc': 0.75
        }
    }

    results = {}
    for event_name, event_config in events.items():
        event_data = load_data_for_period(event_config['start'], event_config['end'])

        # Evaluate model
        eval_results = evaluate_model(model, event_data)

        # Check if meets minimum requirements
        passed = eval_results['auc_roc'] >= event_config['min_auc']

        results[event_name] = {
            'auc': eval_results['auc_roc'],
            'actual_spikes': event_data['price_spike'].sum(),
            'expected_spikes': event_config['expected_spikes'],
            'passed': passed
        }

        print(f"\n{event_name}:")
        print(f"  AUC: {eval_results['auc_roc']:.4f} (min: {event_config['min_auc']})")
        print(f"  Status: {'✅ PASS' if passed else '❌ FAIL'}")

    # Overall pass/fail
    all_passed = all(r['passed'] for r in results.values())

    if all_passed:
        print(f"\n{'='*80}")
        print(f"✅ MODEL PASSED ALL HISTORICAL EVENT TESTS")
        print(f"{'='*80}\n")
    else:
        failed_events = [name for name, r in results.items() if not r['passed']]
        print(f"\n{'='*80}")
        print(f"❌ MODEL FAILED: {', '.join(failed_events)}")
        print(f"{'='*80}\n")

    return results
```

### 5.2 Stress Testing

```python
def stress_test_model(model, test_df):
    """
    Test model under extreme/unusual conditions.
    """

    stress_tests = []

    # Test 1: Extreme prices
    extreme_prices = test_df[test_df['rt_lmp'] > 1000]
    if len(extreme_prices) > 0:
        auc = roc_auc_score(extreme_prices['price_spike'], model.predict(extreme_prices))
        stress_tests.append({'test': 'Extreme Prices (>$1000)', 'auc': auc, 'passed': auc > 0.80})

    # Test 2: Extreme forecast errors
    extreme_errors = test_df[
        (abs(test_df['load_error_pct']) > 5) |
        (abs(test_df['wind_error_pct']) > 20)
    ]
    if len(extreme_errors) > 0:
        auc = roc_auc_score(extreme_errors['price_spike'], model.predict(extreme_errors))
        stress_tests.append({'test': 'Extreme Forecast Errors', 'auc': auc, 'passed': auc > 0.80})

    # Test 3: Rare hours (midnight-4 AM)
    rare_hours = test_df[test_df['timestamp'].dt.hour.isin([0, 1, 2, 3, 4])]
    if len(rare_hours) > 0:
        auc = roc_auc_score(rare_hours['price_spike'], model.predict(rare_hours))
        stress_tests.append({'test': 'Rare Hours (12-4 AM)', 'auc': auc, 'passed': auc > 0.75})

    # Test 4: High renewable penetration
    high_renewables = test_df[
        (test_df['wind_gen_pct'] > 50) | (test_df['solar_gen_pct'] > 30)
    ]
    if len(high_renewables) > 0:
        auc = roc_auc_score(high_renewables['price_spike'], model.predict(high_renewables))
        stress_tests.append({'test': 'High Renewable Penetration', 'auc': auc, 'passed': auc > 0.80})

    print(f"\n{'='*80}")
    print(f"⚡ STRESS TEST RESULTS")
    print(f"{'='*80}\n")

    for test in stress_tests:
        status = '✅ PASS' if test['passed'] else '❌ FAIL'
        print(f"  {test['test']:.<40} AUC: {test['auc']:.4f} {status}")

    all_passed = all(t['passed'] for t in stress_tests)
    print(f"\n{'='*80}")
    print(f"{'✅ ALL STRESS TESTS PASSED' if all_passed else '❌ SOME STRESS TESTS FAILED'}")
    print(f"{'='*80}\n")

    return stress_tests
```

---

## 6. Deployment Checklist

Before deploying any model to production:

### Model Performance
- [ ] AUC > target threshold on validation set
- [ ] AUC > target threshold on test set (completely held out)
- [ ] Passed all historical event backtests
- [ ] Passed all stress tests
- [ ] Cross-model consistency validated
- [ ] Financial impact positive (trading simulation)

### Robustness
- [ ] Stable performance across seasons
- [ ] Stable performance across hours
- [ ] Good performance during extreme weather
- [ ] Handles missing features gracefully
- [ ] No data leakage verified

### Explainability
- [ ] SHAP values computed and validated
- [ ] Feature importance analyzed
- [ ] Failure modes understood
- [ ] Counterfactual analysis available

### Technical
- [ ] Model serialized and versioned
- [ ] Inference latency < 100ms
- [ ] GPU memory usage acceptable
- [ ] Batch inference working
- [ ] Monitoring metrics defined

### Documentation
- [ ] Model card created
- [ ] Training process documented
- [ ] Hyperparameters recorded
- [ ] Known limitations documented
- [ ] Improvement history logged

---

## 7. Continuous Improvement

Even after reaching world-class performance, continue improving:

### Weekly
- Monitor production performance
- Collect new data
- Track model drift

### Monthly
- Retrain with latest data
- Re-evaluate on test set
- Update benchmarks

### Quarterly
- Deep feature engineering review
- Architecture improvements
- Ensemble updates

### Yearly
- Major model architecture overhaul
- Incorporate latest research
- Complete system redesign if needed

---

**Remember: The future of the company depends on these models. Never settle for "good enough". Always push for world-class.**
