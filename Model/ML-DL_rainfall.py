import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Import libraries for models
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
from tqdm import tqdm
import itertools
import os
from datetime import datetime

# ==================== GLOBAL CONFIGURATION ====================
class Config:
    SEQ_LENGTH = 7  # 7 historical days
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    N_SPLITS = 16  # Cross-validation folds
    PATIENCE = 30  # Early stopping

config = Config()

# ==================== STATISTICAL METRICS ====================
class MetricsCalculator:
    """Computes all required statistical metrics"""

    @staticmethod
    def nse(y_true, y_pred):
        """Nash-Sutcliffe Efficiency"""
        numerator = np.sum((y_true - y_pred) ** 2)
        denominator = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (numerator / denominator) if denominator != 0 else float('-inf')

    @staticmethod
    def pearson_r(y_true, y_pred):
        """Pearson correlation coefficient"""
        if len(y_true.flatten()) < 2:
            return 0.0
        return np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1]

    @staticmethod
    def r2_score(y_true, y_pred):
        """Coefficient of determination"""
        return r2_score(y_true, y_pred)

    @staticmethod
    def rmse(y_true, y_pred):
        """Root Mean Square Error"""
        return np.sqrt(mean_squared_error(y_true, y_pred))

    @staticmethod
    def mse(y_true, y_pred):
        """Mean Square Error"""
        return mean_squared_error(y_true, y_pred)

    @staticmethod
    def nrmse(y_true, y_pred):
        """Normalized RMSE"""
        rmse_val = np.sqrt(mean_squared_error(y_true, y_pred))
        range_val = np.max(y_true) - np.min(y_true)
        return rmse_val / range_val if range_val != 0 else rmse_val

    @staticmethod
    def calculate_all_metrics(y_true, y_pred, model_name="", phase=""):
        """Compute all metrics"""
        metrics = {
            'NSE': MetricsCalculator.nse(y_true, y_pred),
            'R': MetricsCalculator.pearson_r(y_true, y_pred),
            'R2': MetricsCalculator.r2_score(y_true, y_pred),
            'RMSE': MetricsCalculator.rmse(y_true, y_pred),
            'MSE': MetricsCalculator.mse(y_true, y_pred),
            'NRMSE': MetricsCalculator.nrmse(y_true, y_pred),
            'MAE': mean_absolute_error(y_true, y_pred)
        }

        if model_name and phase:
            print(f"\n=== {model_name} METRICS - {phase} ===")
            for key, value in metrics.items():
                print(f"{key}: {value:.4f}")

        return metrics

# ==================== RESULTS MANAGER ====================
class ResultsManager:
    """Handles saving and displaying results"""

    def __init__(self, results_dir="results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def save_predictions(self, y_true, y_pred, model_name, phase, station_names=None):
        """Save predictions and observations"""
        if station_names is None:
            station_names = [f"Station_{i+1}" for i in range(y_true.shape[1])]

        results_df = pd.DataFrame()

        for i, station in enumerate(station_names):
            results_df[f'{station}_Observed'] = y_true[:, i]
            results_df[f'{station}_Predicted'] = y_pred[:, i]
            results_df[f'{station}_Error'] = y_true[:, i] - y_pred[:, i]
            results_df[f'{station}_Absolute_Error'] = np.abs(y_true[:, i] - y_pred[:, i])

        filename = f"{self.results_dir}/predictions_{model_name}_{phase}_{self.timestamp}.csv"
        results_df.to_csv(filename, index=False)
        print(f"Predictions saved: {filename}")

        return results_df

    def save_metrics(self, metrics_dict, model_name, phase):
        """Save metrics"""
        metrics_df = pd.DataFrame([metrics_dict])
        filename = f"{self.results_dir}/metrics_{model_name}_{phase}_{self.timestamp}.csv"
        metrics_df.to_csv(filename, index=False)
        print(f"Metrics saved: {filename}")

        return metrics_df

    def save_comparison(self, comparison_df):
        """Save final comparison"""
        filename = f"{self.results_dir}/final_comparison_{self.timestamp}.csv"
        comparison_df.to_csv(filename, index=False)
        print(f"Final comparison saved: {filename}")

        return filename

  def save_lstm_grid_results(self, grid_results, model_name):
    """Save LSTM grid search results"""
    grid_df = pd.DataFrame(grid_results)
    filename = f"{self.results_dir}/lstm_grid_search_{model_name}_{self.timestamp}.csv"
    grid_df.to_csv(filename, index=False)
    print(f"LSTM grid search results saved: {filename}")
    return grid_df


# ==================== DATA PREPROCESSING ====================
class DataPreprocessor:
    """Prepares data for machine learning"""

    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.scaler_x = MinMaxScaler(feature_range=(0, 1))
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))

    def load_and_preprocess_data(self):
        """Load and preprocess data"""
        print("Loading and preprocessing data...")

        # Load dataset
        self.data = pd.read_csv(self.file_path)
        data_selected = self.data.iloc[:, 1:]

        # Column separation
        self.rain_columns = data_selected.iloc[:, 0:10].columns
        self.other_columns = data_selected.iloc[:, 10:].columns

        # Train/test split
        data_train = data_selected.iloc[:11310]
        data_test = data_selected.iloc[11310:]

        # Normalization
        data_train_other = self.scaler_x.fit_transform(data_train[self.other_columns])
        data_train_rain = self.scaler_y.fit_transform(data_train[self.rain_columns])

        data_test_other = self.scaler_x.transform(data_test[self.other_columns])
        data_test_rain = self.scaler_y.transform(data_test[self.rain_columns])

        # Reconstruction
        self.data_train_scaled = np.hstack([data_train_rain, data_train_other])
        self.data_test_scaled = np.hstack([data_test_rain, data_test_other])

        print(f"Training data shape: {self.data_train_scaled.shape}")
        print(f"Test data shape: {self.data_test_scaled.shape}")

        return self.data_train_scaled, self.data_test_scaled

    def create_sequences(self, data, n_steps=7):
        """Create sequences for training and remove NaN values"""
        X, y = [], []

        for i in range(len(data)):
            end_ix = i + n_steps
            if end_ix > len(data) - 1:
                break
            seq_x, seq_y = data[i:end_ix, :], data[end_ix, 0:10]
            X.append(seq_x)
            y.append(seq_y)

        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)

        print(f"Sequences created - X: {X.shape}, y: {y.shape}")

        # Remove NaN sequences
        mask_x = ~np.isnan(X).any(axis=(1, 2))
        mask_y = ~np.isnan(y).any(axis=1)
        mask = mask_x & mask_y

        X_clean = X[mask]
        y_clean = y[mask]

        print(f"After NaN removal - X: {X_clean.shape}, y: {y_clean.shape}")
        print(f"Removed sequences: {len(X) - len(X_clean)}")

        return X_clean, y_clean

    def prepare_datasets(self):
        """Prepare all datasets"""
        train_data, test_data = self.load_and_preprocess_data()

        # Sequence creation with NaN removal
        X_train, y_train = self.create_sequences(train_data, config.SEQ_LENGTH)
        X_test, y_test = self.create_sequences(test_data, config.SEQ_LENGTH)

        # Flattened data for non-sequential models
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)

        print(f"\nFinal datasets:")
        print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
        print(f"X_train_flat: {X_train_flat.shape}")

        return {
            'sequential': {
                'X_train': torch.tensor(X_train).float(),
                'X_test': torch.tensor(X_test).float(),
                'y_train': torch.tensor(y_train).float(),
                'y_test': torch.tensor(y_test).float()
            },
            'flat': {
                'X_train': X_train_flat,
                'X_test': X_test_flat,
                'y_train': y_train,
                'y_test': y_test
            },
            'scalers': {
                'x': self.scaler_x,
                'y': self.scaler_y
            },
            'station_names': [f"Station_{i+1}" for i in range(10)]
        }
      
      # ==================== NEURAL NETWORK MODELS ====================
class LSTM_Model(nn.Module):
    def __init__(self, input_dim=100, n_layers=2, n_hidden=50, output_dim=10, dropout=0.2):
        super(LSTM_Model, self).__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=n_hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(n_hidden, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.n_layers, x.size(0), self.n_hidden).to(x.device)
        c0 = torch.zeros(self.n_layers, x.size(0), self.n_hidden).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])  # Last time step
        out = self.linear(out)
        return out

# ==================== LSTM GRID SEARCH ====================
class LSTMGridSearch:
    """Grid search for LSTM hyperparameters"""

    def __init__(self, input_dim, output_dim=10, results_manager=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.results_manager = results_manager
        self.best_params = None
        self.best_score = float('inf')
        self.grid_results = []

    def generate_param_combinations(self):
        """Generate parameter combinations for grid search"""
        param_grid = {
            'n_layers': [3, 4],
            'n_hidden': [200, 300],
            'dropout': [0, 0.1],
            'learning_rate': [0.001, 0.0001],
            'batch_size': [32, 64]
        }

        keys = param_grid.keys()
        values = param_grid.values()

        combinations = [
            dict(zip(keys, combo))
            for combo in itertools.product(*values)
        ]

        print(f"Number of configurations to test: {len(combinations)}")
        return combinations

    def train_model_with_params(self, params, X_train, y_train, X_val, y_val, device):
        """Train LSTM model with specific hyperparameters"""
        model = LSTM_Model(
            input_dim=self.input_dim,
            n_layers=params['n_layers'],
            n_hidden=params['n_hidden'],
            output_dim=self.output_dim,
            dropout=params['dropout']
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
        criterion = nn.MSELoss()

        # Data loaders
        train_dataset = data.TensorDataset(
            torch.tensor(X_train).float().to(device),
            torch.tensor(y_train).float().to(device)
        )

        val_dataset = data.TensorDataset(
            torch.tensor(X_val).float().to(device),
            torch.tensor(y_val).float().to(device)
        )

        train_loader = data.DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        val_loader = data.DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)

        # Early stopping
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0

        for epoch in range(100):  # Max 100 epochs

            # Training phase
            model.train()
            train_loss = 0.0

            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation phase
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)

            # Early stopping logic
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        return best_val_loss

    def search_best_params(self, X_train, y_train, n_splits=3, max_combinations=None):
        """Run grid search with cross-validation"""
        print("\n STARTING LSTM GRID SEARCH")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        combinations = self.generate_param_combinations()

        # Limit combinations if needed
        if max_combinations is not None and len(combinations) > max_combinations:
            combinations = combinations[:max_combinations]
            print(f"Limited to {max_combinations} combinations for practicality")

        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)

        for i, params in enumerate(combinations):
            print(f"\n🔧 Configuration {i+1}/{len(combinations)}")
            print(f"Params: {params}")

            fold_scores = []

            for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
                print(f"  Fold {fold+1}/{n_splits}...")

                X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
                y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

                val_loss = self.train_model_with_params(
                    params, X_train_fold, y_train_fold, X_val_fold, y_val_fold, device
                )

                fold_scores.append(val_loss)

            avg_score = np.mean(fold_scores)

            self.grid_results.append({
                'params': params,
                'avg_val_loss': avg_score,
                'fold_scores': fold_scores
            })

            print(f"  Average score: {avg_score:.4f}")

            # Update best model
            if avg_score < self.best_score:
                self.best_score = avg_score
                self.best_params = params
                print(f" NEW BEST SCORE: {avg_score:.4f}")

        # Sort results
        self.grid_results.sort(key=lambda x: x['avg_val_loss'])

        # Display best results
        print("\n" + "="*60)
        print("BEST LSTM GRID SEARCH RESULTS")
        print("="*60)

        for i, result in enumerate(self.grid_results[:5]):
            print(f"{i+1}. Score: {result['avg_val_loss']:.4f}")
            print(f"   Params: {result['params']}\n")

        # Save results
        if self.results_manager:
            self.results_manager.save_lstm_grid_results(
                self.grid_results,
                "LSTM_GridSearch"
            )
        return self.best_params, self.grid_results
      
    # ==================== CROSS-VALIDATION FOR ML MODELS ====================
class MLCrossValidation:
    """Cross-validation for classical ML models"""

    def __init__(self, results_manager):
        self.results_manager = results_manager
        self.best_models = {}
        self.cv_scores = {}

    def run_cross_validation(self, model, model_name, X_train, y_train, scaler_y, station_names, n_splits=16):
        """Performs cross-validation for a ML model"""
        print(f"\n CROSS-VALIDATION {model_name} ({n_splits} folds)")

        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)
        fold_scores = {'train': [], 'val': []}
        fold_predictions = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
            print(f"\n=== Fold {fold + 1}/{n_splits} ===")

            # Data split
            X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
            y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

            # Model training
            model.fit(X_train_fold, y_train_fold)

            # Predictions on train and validation sets
            y_train_pred_scaled = model.predict(X_train_fold)
            y_val_pred_scaled = model.predict(X_val_fold)

            # Inverse transformation
            y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled)
            y_train_true = scaler_y.inverse_transform(y_train_fold)
            y_val_pred = scaler_y.inverse_transform(y_val_pred_scaled)
            y_val_true = scaler_y.inverse_transform(y_val_fold)

            # Metrics calculation
            train_metrics = MetricsCalculator.calculate_all_metrics(y_train_true, y_train_pred)
            val_metrics = MetricsCalculator.calculate_all_metrics(y_val_true, y_val_pred)

            fold_scores['train'].append(train_metrics['R2'])
            fold_scores['val'].append(val_metrics['R2'])

            print(f"R² Train: {train_metrics['R2']:.4f}, R² Val: {val_metrics['R2']:.4f}")

            # Save fold predictions
            fold_pred_df = pd.DataFrame({
                'fold': fold,
                'y_true': y_val_true.flatten(),
                'y_pred': y_val_pred.flatten()
            })
            fold_predictions.append(fold_pred_df)

        # Mean scores
        mean_train_score = np.mean(fold_scores['train'])
        mean_val_score = np.mean(fold_scores['val'])
        std_val_score = np.std(fold_scores['val'])

        print(f"\n📊 CROSS-VALIDATION RESULTS {model_name}:")
        print(f"  Mean R² (train): {mean_train_score:.4f}")
        print(f"  Mean R² (val): {mean_val_score:.4f}")
        print(f"  Std (val): {std_val_score:.4f}")

        self.cv_scores[model_name] = {
            'mean_train_score': mean_train_score,
            'mean_val_score': mean_val_score,
            'std_val_score': std_val_score,
            'fold_scores': fold_scores
        }

        return mean_val_score

    def run_cross_validation_with_grid_search(self, model_class, model_name, X_train, y_train, param_grid,
                                              scaler_y, station_names, n_splits=16):
        """Cross-validation with hyperparameter search"""
        print(f"\n CROSS-VALIDATION WITH GRID SEARCH {model_name}")

        # Model creation with MultiOutputRegressor
        if model_name in ['Random Forest', 'Extra Trees', 'XGBoost']:
            base_model = model_class(random_state=config.RANDOM_STATE)
            model = MultiOutputRegressor(base_model)
        else:
            model = model_class(random_state=config.RANDOM_STATE)

        # Randomized grid search with cross-validation
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=10,
            cv=n_splits,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            random_state=config.RANDOM_STATE
        )

        search.fit(X_train, y_train)

        print(f"Best parameters {model_name}: {search.best_params_}")
        print(f"Best score: {-search.best_score_:.4f}")

        # Final model with best parameters
        best_model = search.best_estimator_

        # Cross-validation with best model
        cv_score = self.run_cross_validation(
            best_model, model_name, X_train, y_train,
            scaler_y, station_names, n_splits
        )

        self.best_models[model_name] = best_model

        return best_model, cv_score
# ==================== TRAINERS WITH FULL EVALUATION ====================
class LSTMTrainer:
    """LSTM training with cross-validation and full evaluation"""

    def __init__(self, model, device, results_manager):
        self.model = model
        self.device = device
        self.results_manager = results_manager
        self.best_model_state = None

    def train_with_cross_validation(self, X_train, y_train, X_test, y_test, scaler_y,
                                    station_names, n_splits=16, patience=30, epochs=500):
        """Training with cross-validation and full evaluation"""
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
            print(f"\n=== Fold {fold + 1}/{n_splits} ===")

            # Data split
            X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
            y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

            # DataLoaders creation
            train_dataset = data.TensorDataset(
                torch.tensor(X_train_fold).float().to(self.device),
                torch.tensor(y_train_fold).float().to(self.device)
            )
            val_dataset = data.TensorDataset(
                torch.tensor(X_val_fold).float().to(self.device),
                torch.tensor(y_val_fold).float().to(self.device)
            )

            train_loader = data.DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = data.DataLoader(val_dataset, batch_size=32, shuffle=False)

            # Train fold
            fold_metrics = self._train_fold(train_loader, val_loader, patience, epochs, fold)
            fold_results.append(fold_metrics)

        # Evaluation on full training set
        print("\n=== TRAINING SET EVALUATION ===")
        train_predictions, train_metrics = self.evaluate_model(
            X_train, y_train, scaler_y, station_names, "LSTM", "Training"
        )

        # Evaluation on test set
        print("\n=== TEST SET EVALUATION ===")
        test_predictions, test_metrics = self.evaluate_model(
            X_test, y_test, scaler_y, station_names, "LSTM", "Test"
        )

        return {
            'fold_results': fold_results,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'train_predictions': train_predictions,
            'test_predictions': test_predictions
        }

    def _train_fold(self, train_loader, val_loader, patience, epochs, fold):
        """Training for a single fold"""
        optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=64)
        criterion = nn.MSELoss()

        best_val_loss = float('inf')
        patience_counter = 0
        fold_history = {'train_loss': [], 'val_loss': []}

        for epoch in range(epochs):

            # Training phase
            self.model.train()
            train_loss = 0.0

            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)

            fold_history['train_loss'].append(avg_train_loss)
            fold_history['val_loss'].append(avg_val_loss)

            scheduler.step()

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                self.best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        return {'best_val_loss': best_val_loss, 'history': fold_history}

    def evaluate_model(self, X, y, scaler_y, station_names, model_name, phase):
        """Full model evaluation"""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X).float().to(self.device)
            y_pred_scaled = self.model(X_tensor).cpu().numpy()

        # Inverse transformation
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        y_true = scaler_y.inverse_transform(y)

        # Metrics computation
        metrics = MetricsCalculator.calculate_all_metrics(y_true, y_pred, model_name, phase)

        # Save predictions
        predictions_df = self.results_manager.save_predictions(
            y_true, y_pred, model_name, phase, station_names
        )

        # Save metrics
        self.results_manager.save_metrics(metrics, model_name, phase)

        return predictions_df, metrics
    class MLPTrainer:
    """MLP training with full evaluation"""

    def __init__(self, results_manager):
        self.best_model = None
        self.best_params = None
        self.results_manager = results_manager

    def train_with_grid_search(self, X_train, y_train, X_test, y_test, scaler_y, station_names):
        """MLP training with hyperparameter search and full evaluation"""
        print("\n=== MLP HYPERPARAMETER SEARCH ===")

        # Correct parameter grid for MLPRegressor
        param_grid = {
            'hidden_layer_sizes': [(100,), (100, 50)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001],
            'learning_rate_init': [0.001, 0.01],
            'batch_size': [32, 64],
            'max_iter': [500]
        }

        mlp = MLPRegressor(
            random_state=config.RANDOM_STATE,
            early_stopping=True
        )

        # Randomized grid search with cross-validation
        grid_search = RandomizedSearchCV(
            estimator=mlp,
            param_distributions=param_grid,
            n_iter=4,
            cv=config.N_SPLITS,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            random_state=config.RANDOM_STATE
        )

        grid_search.fit(X_train, y_train)

        self.best_params = grid_search.best_params_
        self.best_model = grid_search.best_estimator_

        print(f"Best MLP parameters: {grid_search.best_params_}")
        print(f"Best score: {-grid_search.best_score_:.4f}")

        # Full evaluation
        train_results = self.evaluate_model(
            self.best_model, X_train, y_train, scaler_y, station_names, "MLP", "Training"
        )

        test_results = self.evaluate_model(
            self.best_model, X_test, y_test, scaler_y, station_names, "MLP", "Test"
        )

        return {
            'model': self.best_model,
            'params': self.best_params,
            'train_metrics': train_results['metrics'],
            'test_metrics': test_results['metrics'],
            'train_predictions': train_results['predictions'],
            'test_predictions': test_results['predictions']
        }

    def evaluate_model(self, model, X, y, scaler_y, station_names, model_name, phase):
        """Full model evaluation"""
        # Predictions
        y_pred_scaled = model.predict(X)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        y_true = scaler_y.inverse_transform(y)

        # Metrics computation
        metrics = MetricsCalculator.calculate_all_metrics(
            y_true, y_pred, model_name, phase
        )

        # Save predictions
        predictions_df = self.results_manager.save_predictions(
            y_true, y_pred, model_name, phase, station_names
        )

        # Save metrics
        self.results_manager.save_metrics(metrics, model_name, phase)

        return {'metrics': metrics, 'predictions': predictions_df}


# ==================== ML MODELS WITH CROSS-VALIDATION ====================
class MLModelsTrainer:
    """Machine Learning models with cross-validation and hyperparameter tuning"""

    def __init__(self, results_manager):
        self.models = {}
        self.best_params = {}
        self.results_manager = results_manager
        self.ml_cv = MLCrossValidation(results_manager)

    def initialize_models(self):
        """Initialize all ML models with their hyperparameter grids"""

        # Random Forest parameter grid
        rf_param_grid = {
            'estimator__n_estimators': [100, 200, 300],
            'estimator__max_depth': [10, 15, 20, None],
            'estimator__min_samples_split': [2, 5, 10],
            'estimator__min_samples_leaf': [1, 2, 4],
            'estimator__max_features': ['sqrt', 'log2']
        }

        # Extra Trees parameter grid
        et_param_grid = {
            'estimator__n_estimators': [100, 200, 300],
            'estimator__max_depth': [10, 15, 20, None],
            'estimator__min_samples_split': [2, 5, 10],
            'estimator__min_samples_leaf': [1, 2, 4],
            'estimator__max_features': ['sqrt', 'log2']
        }

        # XGBoost parameter grid
        xgb_param_grid = {
            'estimator__n_estimators': [100, 200, 300],
            'estimator__max_depth': [3, 6, 9],
            'estimator__learning_rate': [0.01, 0.1, 0.2],
            'estimator__subsample': [0.8, 0.9, 1.0],
            'estimator__colsample_bytree': [0.8, 0.9, 1.0]
        }

        self.models = {
            'Random Forest': {
                'model_class': RandomForestRegressor,
                'param_grid': rf_param_grid
            },
            'Extra Trees': {
                'model_class': ExtraTreesRegressor,
                'param_grid': et_param_grid
            },
            'XGBoost': {
                'model_class': XGBRegressor,
                'param_grid': xgb_param_grid
            }
        }
        return self.models
  def train_with_cross_validation(self, X_train, y_train, X_test, y_test, scaler_y, station_names):
    """Training with cross-validation and hyperparameter optimization"""
    results = {}

    for name, model_info in self.models.items():
        print(f"\n{'='*60}")
        print(f"TRAINING {name} WITH CROSS-VALIDATION")
        print(f"{'='*60}")

        # Cross-validation with hyperparameter search
        best_model, cv_score = self.ml_cv.run_cross_validation_with_grid_search(
            model_info['model_class'],
            name,
            X_train, y_train,
            model_info['param_grid'],
            scaler_y,
            station_names,
            n_splits=config.N_SPLITS
        )

        # Full evaluation on training set
        train_results = self.evaluate_model(
            best_model, X_train, y_train, scaler_y, station_names, name, "Training"
        )

        # Full evaluation on test set
        test_results = self.evaluate_model(
            best_model, X_test, y_test, scaler_y, station_names, name, "Test"
        )

        results[name] = {
            'best_model': best_model,
            'cv_score': cv_score,
            'cv_scores': self.ml_cv.cv_scores[name],
            'train_metrics': train_results['metrics'],
            'test_metrics': test_results['metrics'],
            'train_predictions': train_results['predictions'],
            'test_predictions': test_results['predictions']
        }

        print(f"\n📊 FINAL PERFORMANCE {name}:")
        print(f"  Training R²: {train_results['metrics']['R2']:.4f}")
        print(f"  Test R²: {test_results['metrics']['R2']:.4f}")
        print(f"  Test RMSE: {test_results['metrics']['RMSE']:.4f}")

    return results


def evaluate_model(self, model, X, y, scaler_y, station_names, model_name, phase):
    """Full model evaluation"""
    # Predictions
    y_pred_scaled = model.predict(X)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(y)

    # Metrics computation
    metrics = MetricsCalculator.calculate_all_metrics(
        y_true, y_pred, model_name, phase
    )

    # Save predictions
    predictions_df = self.results_manager.save_predictions(
        y_true, y_pred, model_name, phase, station_names
    )

    # Save metrics
    self.results_manager.save_metrics(metrics, model_name, phase)

    return {'metrics': metrics, 'predictions': predictions_df}


# ==================== IMPROVED VISUALIZATION ====================
class ResultVisualizer:
    """Enhanced visualization of model results"""

    @staticmethod
    def plot_predictions(y_true, y_pred, title, station_idx=0):
        """Plot observed vs predicted values"""
        plt.figure(figsize=(15, 6))

        plt.subplot(1, 2, 1)
        plt.plot(y_true[:, station_idx], 'b-', label='Observed', alpha=0.7, linewidth=1)
        plt.plot(y_pred[:, station_idx], 'r-', label='Predicted', alpha=0.7, linewidth=1)
        plt.title(f'{title} - Station {station_idx + 1}')
        plt.xlabel('Time')
        plt.ylabel('Precipitation (mm)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.scatter(y_true[:, station_idx], y_pred[:, station_idx], alpha=0.6)

        min_val = min(y_true[:, station_idx].min(), y_pred[:, station_idx].min())
        max_val = max(y_true[:, station_idx].max(), y_pred[:, station_idx].max())

        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8)
        plt.xlabel('Observed')
        plt.ylabel('Predicted')
        plt.title(f'Observed vs Predicted - Station {station_idx + 1}')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_metrics_comparison(metrics_dict_train, metrics_dict_test):
        """Comparison of metrics between models for training and test sets"""
        models = list(metrics_dict_train.keys())
        metrics_names = ['R2', 'NSE', 'RMSE']

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))

        # Training metrics
        for i, metric in enumerate(metrics_names):
            values = [metrics_dict_train[model][metric] for model in models]
            axes[0, i].bar(models, values, color=colors)
            axes[0, i].set_title(f'Training - {metric}')
            axes[0, i].set_ylabel(metric)
            axes[0, i].tick_params(axis='x', rotation=45)

            for j, v in enumerate(values):
                axes[0, i].text(j, v + 0.01 * max(values), f'{v:.3f}',
                                ha='center', va='bottom', fontweight='bold')

        # Test metrics
        for i, metric in enumerate(metrics_names):
            values = [metrics_dict_test[model][metric] for model in models]
            axes[1, i].bar(models, values, color=colors)
            axes[1, i].set_title(f'Test - {metric}')
            axes[1, i].set_ylabel(metric)
            axes[1, i].tick_params(axis='x', rotation=45)

            for j, v in enumerate(values):
                axes[1, i].text(j, v + 0.01 * max(values), f'{v:.3f}',
                                ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.show()
    def save_lstm_grid_results(self, grid_results, model_name):
    """Save LSTM grid search results"""
    grid_df = pd.DataFrame(grid_results)
    filename = f"{self.results_dir}/lstm_grid_search_{model_name}_{self.timestamp}.csv"
    grid_df.to_csv(filename, index=False)
    print(f"LSTM grid search results saved: {filename}")
    return grid_df

# ==================== DATA PREPARATION ====================
class DataPreprocessor:
    """Prepares data for machine learning"""

    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.scaler_x = MinMaxScaler(feature_range=(0, 1))
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))

    def load_and_preprocess_data(self):
        """Loads and preprocesses data"""
        print("Loading and preprocessing data...")

        # Loading
        self.data = pd.read_csv(self.file_path)
        data_selected = self.data.iloc[:, 1:]

        # Column separation
        self.col_pluie = data_selected.iloc[:, 0:10].columns
        self.autre_col = data_selected.iloc[:, 10:].columns

        # Train/test split
        data_train = data_selected.iloc[:11310]
        data_test = data_selected.iloc[11310:]

        # Normalization
        data_train_autre = self.scaler_x.fit_transform(data_train[self.autre_col])
        data_train_pluie = self.scaler_y.fit_transform(data_train[self.col_pluie])

        data_test_autre = self.scaler_x.transform(data_test[self.autre_col])
        data_test_pluie = self.scaler_y.transform(data_test[self.col_pluie])

        # Reconstruction
        self.data_train_scaled = np.hstack([data_train_pluie, data_train_autre])
        self.data_test_scaled = np.hstack([data_test_pluie, data_test_autre])

        print(f"Training data: {self.data_train_scaled.shape}")
        print(f"Test data: {self.data_test_scaled.shape}")

        return self.data_train_scaled, self.data_test_scaled

    def create_sequences(self, data, n_steps=7):
        """Creates sequences for learning and removes NaNs"""
        X, y = [], []
        for i in range(len(data)):
            end_ix = i + n_steps
            if end_ix > len(data) - 1:
                break
            seq_x, seq_y = data[i:end_ix, :], data[end_ix, 0:10]
            X.append(seq_x)
            y.append(seq_y)

        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)

        print(f"Sequences created - X: {X.shape}, y: {y.shape}")

        # Remove sequences containing NaNs
        mask_x = ~np.isnan(X).any(axis=(1, 2))
        mask_y = ~np.isnan(y).any(axis=1)
        mask = mask_x & mask_y

        X_clean = X[mask]
        y_clean = y[mask]

        print(f"After NaN removal - X: {X_clean.shape}, y: {y_clean.shape}")
        print(f"Removed sequences: {len(X) - len(X_clean)}")

        return X_clean, y_clean

    def prepare_datasets(self):
        """Prepares all datasets"""
        train_data, test_data = self.load_and_preprocess_data()

        # Sequence creation with NaN removal
        X_train, y_train = self.create_sequences(train_data, config.SEQ_LENGTH)
        X_test, y_test = self.create_sequences(test_data, config.SEQ_LENGTH)

        # For non-sequential models (reshape)
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)

        print(f"\nFinal datasets:")
        print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
        print(f"X_train_flat: {X_train_flat.shape}")

        return {
            'sequential': {
                'X_train': torch.tensor(X_train).float(),
                'X_test': torch.tensor(X_test).float(),
                'y_train': torch.tensor(y_train).float(),
                'y_test': torch.tensor(y_test).float()
            },
            'flat': {
                'X_train': X_train_flat,
                'X_test': X_test_flat,
                'y_train': y_train,
                'y_test': y_test
            },
            'scalers': {
                'x': self.scaler_x,
                'y': self.scaler_y
            },
            'station_names': [f"Station_{i+1}" for i in range(10)]
        }
    # ==================== NEURAL NETWORK MODELS ====================
class LSTM_Model(nn.Module):
    def __init__(self, input_dim=100, n_layers=2, n_hidden=50, output_dim=10, dropout=0.2):
        super(LSTM_Model, self).__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=n_hidden,
                            num_layers=n_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(n_hidden, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.n_layers, x.size(0), self.n_hidden).to(x.device)
        c0 = torch.zeros(self.n_layers, x.size(0), self.n_hidden).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])  # Last time step
        out = self.linear(out)
        return out
# ==================== LSTM GRID SEARCH ====================
class LSTMGridSearch:
    """Grid search for LSTM hyperparameters"""

    def __init__(self, input_dim, output_dim=10, results_manager=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.results_manager = results_manager
        self.best_params = None
        self.best_score = float('inf')
        self.grid_results = []

    def generate_param_combinations(self):
        """Generates parameter combinations for grid search"""
        param_grid = {
            'n_layers': [3, 4],
            'n_hidden': [200, 300],
            'dropout': [0, 0.1],
            'learning_rate': [0.001, 0.0001],
            'batch_size': [32, 64]
        }

        keys = param_grid.keys()
        values = param_grid.values()
        combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

        print(f"Number of combinations to test: {len(combinations)}")
        return combinations

    def search_best_params(self, X_train, y_train, n_splits=3, max_combinations=None):
        print("\n STARTING LSTM GRID SEARCH")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        combinations = self.generate_param_combinations()

        if max_combinations is not None and len(combinations) > max_combinations:
            combinations = combinations[:max_combinations]
            print(f"Limiting to {max_combinations} combinations")

        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)

        for i, params in enumerate(combinations):
            print(f"\n Configuration {i+1}/{len(combinations)}")
            print(f"Params: {params}")

            fold_scores = []

            for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
                print(f"  Fold {fold+1}/{n_splits}...")

                X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
                y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

                val_loss = self.train_model_with_params(
                    params, X_train_fold, y_train_fold, X_val_fold, y_val_fold, device
                )
                fold_scores.append(val_loss)

            avg_score = np.mean(fold_scores)
            self.grid_results.append({
                'params': params,
                'avg_val_loss': avg_score,
                'fold_scores': fold_scores
            })

            print(f"  Mean score: {avg_score:.4f}")

            if avg_score < self.best_score:
                self.best_score = avg_score
                self.best_params = params
                print(f" NEW BEST SCORE: {avg_score:.4f}")

        self.grid_results.sort(key=lambda x: x['avg_val_loss'])

        print("\n" + "="*60)
        print("BEST LSTM GRID SEARCH RESULTS")
        print("="*60)

        if self.results_manager:
            self.results_manager.save_lstm_grid_results(self.grid_results, "LSTM_GridSearch")

        return self.best_params, self.grid_results
  # ==================== CROSS VALIDATION FOR ML MODELS ====================
class MLCrossValidation:
    """Cross validation for classical ML models"""

    def run_cross_validation(self, model, model_name, X_train, y_train, scaler_y, station_names, n_splits=16):
        print(f"\n CROSS VALIDATION {model_name} ({n_splits} folds)")

        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)

        fold_scores = {'train': [], 'val': []}

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
            print(f"\n=== Fold {fold + 1}/{n_splits} ===")

            X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
            y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

            model.fit(X_train_fold, y_train_fold)

            y_train_pred_scaled = model.predict(X_train_fold)
            y_val_pred_scaled = model.predict(X_val_fold)

            y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled)
            y_train_true = scaler_y.inverse_transform(y_train_fold)

            y_val_pred = scaler_y.inverse_transform(y_val_pred_scaled)
            y_val_true = scaler_y.inverse_transform(y_val_fold)

            train_metrics = MetricsCalculator.calculate_all_metrics(y_train_true, y_train_pred)
            val_metrics = MetricsCalculator.calculate_all_metrics(y_val_true, y_val_pred)

            fold_scores['train'].append(train_metrics['R2'])
            fold_scores['val'].append(val_metrics['R2'])

            print(f"Train R²: {train_metrics['R2']:.4f}, Val R²: {val_metrics['R2']:.4f}")

        mean_train = np.mean(fold_scores['train'])
        mean_val = np.mean(fold_scores['val'])
        std_val = np.std(fold_scores['val'])

        print(f"\n CROSS VALIDATION RESULTS {model_name}:")
        print(f"  Mean Train R²: {mean_train:.4f}")
        print(f"  Mean Val R²: {mean_val:.4f}")
        print(f"  Val Std: {std_val:.4f}")

        return mean_val
      
      # ==================== MAIN PIPELINE ====================
class PrecipitationForecastPipeline:
    """Complete pipeline for precipitation forecasting"""

    def run_complete_pipeline(self, use_lstm_grid_search=False):
        print("=== STARTING PRECIPITATION FORECAST PIPELINE ===")

        # 1. Data preparation
        self.datasets = self.data_processor.prepare_datasets()

        # 2. LSTM training with optional grid search
        lstm_results = self.run_lstm_model(use_grid_search=use_lstm_grid_search)

        # 3. MLP training (optional)
        mlp_results = self.run_mlp_model()

        # 4. ML models training with cross validation
        ml_results = self.run_ml_models()

        # 5. Results comparison
        self._compare_results()

        # 6. Visualizations
        self._create_visualizations()

        return self.results

  def _compare_results(self):
    """Compares the performance of all models"""
    print("\n" + "="*60)
    print("FINAL MODEL COMPARISON")
    print("="*60)

    comparison_data = []
    for model_name, result in self.results.items():
        train_metrics = result['train_metrics']
        test_metrics = result['test_metrics']

        # Add cross-validation scores if available
        cv_score = ""
        if 'cv_scores' in result:
            cv_score = f"{result['cv_scores']['mean_val_score']:.4f} ± {result['cv_scores']['std_val_score']:.4f}"

        # Training metrics
        comparison_data.append({
            'Model': model_name,
            'Phase': 'Training',
            'R2': f"{train_metrics['R2']:.4f}",
            'NSE': f"{train_metrics['NSE']:.4f}",
            'R': f"{train_metrics['R']:.4f}",
            'RMSE': f"{train_metrics['RMSE']:.4f}",
            'MSE': f"{train_metrics['MSE']:.4f}",
            'NRMSE': f"{train_metrics['NRMSE']:.4f}",
            'MAE': f"{train_metrics['MAE']:.4f}",
            'CV_Score': cv_score
        })

        # Test metrics
        comparison_data.append({
            'Model': model_name,
            'Phase': 'Test',
            'R2': f"{test_metrics['R2']:.4f}",
            'NSE': f"{test_metrics['NSE']:.4f}",
            'R': f"{test_metrics['R']:.4f}",
            'RMSE': f"{test_metrics['RMSE']:.4f}",
            'MSE': f"{test_metrics['MSE']:.4f}",
            'NRMSE': f"{test_metrics['NRMSE']:.4f}",
            'MAE': f"{test_metrics['MAE']:.4f}",
            'CV_Score': cv_score
        })

    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))

    # Save comparison
    self.results_manager.save_comparison(comparison_df)

    # Find best model (based on test R²)
    best_model = max(self.results.items(),
                     key=lambda x: x[1]['test_metrics']['R2'])

    print(f"\n BEST MODEL: {best_model[0]} (Test R² = {best_model[1]['test_metrics']['R2']:.4f})")

    return comparison_df


def _create_visualizations(self):
    """Creates final visualizations"""
    print("\n" + "="*50)
    print("CREATING VISUALIZATIONS")
    print("="*50)

    # Prepare data for plots
    metrics_dict_train = {name: result['train_metrics'] for name, result in self.results.items()}
    metrics_dict_test = {name: result['test_metrics'] for name, result in self.results.items()}

    # Metrics comparison plot
    ResultVisualizer.plot_metrics_comparison(metrics_dict_train, metrics_dict_test)

    # Prediction plots for each model (test only)
    for model_name, result in self.results.items():
        test_true = result['test_predictions'][
            [col for col in result['test_predictions'].columns if '_Observe' in col]
        ].values

        test_pred = result['test_predictions'][
            [col for col in result['test_predictions'].columns if '_Predit' in col]
        ].values

        ResultVisualizer.plot_predictions(
            test_true,
            test_pred,
            f"Test Predictions - {model_name}",
            station_idx=8  # Station 9 (index 8)
        )


# ==================== EXECUTION ====================
if __name__ == "__main__":
    # Data file configuration
    file_path = 'Data/Data_toutes_stations.csv'  # Data

    # Create and run pipeline
    pipeline = PrecipitationForecastPipeline(file_path)

    # Option: use grid search for LSTM
    use_grid_search = False  # Set True to activate LSTM grid search

    results = pipeline.run_complete_pipeline(use_lstm_grid_search=use_grid_search)

    # Final saving
    print("\n FINAL MODEL SAVING")

    for model_name, result in results.items():
        if model_name == 'LSTM' and isinstance(result['model'], nn.Module):
            torch.save(result['model'].state_dict(), f'model_{model_name}.pth')
        elif 'best_model' in result:
            joblib.dump(result['best_model'], f'model_{model_name}.pkl')

    print("Models saved")
    print("\n Pipeline successfully completed!")
