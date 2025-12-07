"""
Distribution Predictor for DDG-DA (Data Distribution Generation for Predictable Concept Drift Adaptation).
Predicts future market regimes based on historical regime statistics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.preprocessing import StandardScaler
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RegimeExtractor:
    """Extract regime features from time series data."""

    def __init__(self, window_size: int = 20):
        """
        Initialize regime extractor.

        Args:
            window_size: Number of days to compute regime statistics
        """
        self.window_size = window_size
        self.scaler = StandardScaler()
        self.fitted = False

    def extract_regime_features(self, df: pd.DataFrame, ticker: str = None) -> pd.DataFrame:
        """
        Extract regime features from price/sentiment data.

        Args:
            df: DataFrame with price and sentiment data
            ticker: Specific ticker to extract (None for all)

        Returns:
            DataFrame with regime features per time period
        """
        if ticker:
            df = df[df['Ticker'] == ticker].copy()

        df = df.sort_values('Date').copy()

        # Price-based regime features
        if 'Close' in df.columns:
            df['price_return'] = df.groupby('Ticker')['Close'].pct_change()
            df['price_volatility'] = df.groupby('Ticker')['price_return'].transform(
                lambda x: x.rolling(self.window_size).std()
            )
            df['price_mean_return'] = df.groupby('Ticker')['price_return'].transform(
                lambda x: x.rolling(self.window_size).mean()
            )
            df['price_skewness'] = df.groupby('Ticker')['price_return'].transform(
                lambda x: x.rolling(self.window_size).skew()
            )
            df['price_kurtosis'] = df.groupby('Ticker')['price_return'].transform(
                lambda x: x.rolling(self.window_size).kurt()
            )

        # Volume-based regime features
        if 'Volume' in df.columns:
            df['volume_mean'] = df.groupby('Ticker')['Volume'].transform(
                lambda x: x.rolling(self.window_size).mean()
            )
            df['volume_std'] = df.groupby('Ticker')['Volume'].transform(
                lambda x: x.rolling(self.window_size).std()
            )
            df['volume_trend'] = df.groupby('Ticker')['Volume'].transform(
                lambda x: x.rolling(self.window_size).apply(
                    lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) == self.window_size else np.nan
                )
            )

        # Sentiment-based regime features (if available)
        sentiment_cols = [col for col in df.columns if 'sentiment' in col.lower()]
        for col in sentiment_cols:
            df[f'{col}_mean'] = df.groupby('Ticker')[col].transform(
                lambda x: x.rolling(self.window_size).mean()
            )
            df[f'{col}_std'] = df.groupby('Ticker')[col].transform(
                lambda x: x.rolling(self.window_size).std()
            )

        # Technical indicator regimes
        if 'rsi' in df.columns:
            df['rsi_mean'] = df.groupby('Ticker')['rsi'].transform(
                lambda x: x.rolling(self.window_size).mean()
            )

        if 'volatility_10d' in df.columns:
            df['realized_vol_mean'] = df.groupby('Ticker')['volatility_10d'].transform(
                lambda x: x.rolling(self.window_size).mean()
            )

        # Market regime indicator (bullish/bearish/neutral)
        if 'price_mean_return' in df.columns and 'price_volatility' in df.columns:
            df['sharpe_ratio'] = df['price_mean_return'] / (df['price_volatility'] + 1e-8)
            df['regime_score'] = df['sharpe_ratio'].rolling(self.window_size).mean()

        return df

    def create_regime_windows(self, df: pd.DataFrame, window_duration: int = 60) -> List[Dict]:
        """
        Divide time series into regime windows (tasks).

        Args:
            df: DataFrame with regime features
            window_duration: Number of days per window

        Returns:
            List of regime dictionaries
        """
        regimes = []

        df = df.sort_values('Date').reset_index(drop=True)

        # Define regime feature columns
        regime_cols = [col for col in df.columns if any(
            keyword in col for keyword in ['volatility', 'return', 'sentiment', 'volume', 'regime', 'sharpe']
        )]

        # Create non-overlapping windows
        for i in range(0, len(df) - window_duration, window_duration // 2):  # 50% overlap
            window_df = df.iloc[i:i + window_duration]

            if len(window_df) < window_duration:
                continue

            # Compute aggregate statistics for this window
            regime_stats = {}
            for col in regime_cols:
                if col in window_df.columns:
                    regime_stats[f'{col}_mean'] = window_df[col].mean()
                    regime_stats[f'{col}_std'] = window_df[col].std()
                    regime_stats[f'{col}_min'] = window_df[col].min()
                    regime_stats[f'{col}_max'] = window_df[col].max()

            regime = {
                'start_date': window_df['Date'].iloc[0],
                'end_date': window_df['Date'].iloc[-1],
                'window_idx': len(regimes),
                'features': regime_stats,
                'data_indices': window_df.index.tolist()
            }

            regimes.append(regime)

        logger.info(f"Created {len(regimes)} regime windows")
        return regimes

    def fit_scaler(self, regimes: List[Dict]):
        """Fit scaler on regime features."""
        feature_vectors = [list(r['features'].values()) for r in regimes]
        self.scaler.fit(feature_vectors)
        self.fitted = True
        logger.info("Regime feature scaler fitted")

    def transform_regime(self, regime: Dict) -> np.ndarray:
        """Transform regime features using fitted scaler."""
        if not self.fitted:
            raise ValueError("Scaler not fitted. Call fit_scaler first.")

        feature_vector = list(regime['features'].values())
        return self.scaler.transform([feature_vector])[0]


class RegimePredictor(nn.Module):
    """Neural network to predict future regime features."""

    def __init__(
            self,
            input_dim: int,
            hidden_dims: List[int] = [128, 64, 32],
            output_dim: int = None,
            dropout: float = 0.3
    ):
        """
        Initialize regime predictor network.

        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension (defaults to input_dim)
            dropout: Dropout rate
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim or input_dim

        # Build MLP
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, self.output_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input regime features [batch_size, seq_len, input_dim] or [batch_size, input_dim]

        Returns:
            Predicted future regime features
        """
        # Handle sequence input
        if x.dim() == 3:
            batch_size, seq_len, _ = x.shape
            x = x.reshape(batch_size * seq_len, -1)
            out = self.network(x)
            out = out.reshape(batch_size, seq_len, -1)
        else:
            out = self.network(x)

        return out


class DDGDADistributionPredictor:
    """Complete DDG-DA distribution predictor with training and inference."""

    def __init__(
            self,
            regime_window_size: int = 20,
            regime_duration: int = 60,
            history_length: int = 5,
            device: str = None
    ):
        """
        Initialize DDG-DA predictor.

        Args:
            regime_window_size: Days to compute regime statistics
            regime_duration: Days per regime window
            history_length: Number of past regimes to use for prediction
            device: Device for training
        """
        self.regime_extractor = RegimeExtractor(window_size=regime_window_size)
        self.regime_duration = regime_duration
        self.history_length = history_length
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = None
        self.regimes = None
        self.feature_dim = None

    def extract_regimes(self, df: pd.DataFrame) -> List[Dict]:
        """Extract regime windows from data."""
        logger.info("Extracting regime features...")

        # Extract regime features
        df_with_regimes = self.regime_extractor.extract_regime_features(df)

        # Create regime windows
        regimes = self.regime_extractor.create_regime_windows(
            df_with_regimes,
            window_duration=self.regime_duration
        )

        self.regimes = regimes

        # Fit scaler
        self.regime_extractor.fit_scaler(regimes)

        # Determine feature dimension
        if regimes:
            self.feature_dim = len(regimes[0]['features'])
            logger.info(f"Regime feature dimension: {self.feature_dim}")

        return regimes

    def create_training_sequences(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create training sequences for regime prediction.

        Returns:
            Tuple of (X, y) where X is historical regimes and y is next regime
        """
        if not self.regimes:
            raise ValueError("No regimes extracted. Call extract_regimes first.")

        X, y = [], []

        # Transform all regimes
        transformed_regimes = [
            self.regime_extractor.transform_regime(r) for r in self.regimes
        ]

        # Create sequences: [r_t-h, ..., r_t-1] -> r_t
        for i in range(self.history_length, len(transformed_regimes)):
            history = transformed_regimes[i - self.history_length:i]
            target = transformed_regimes[i]

            X.append(np.stack(history))
            y.append(target)

        X = np.array(X)
        y = np.array(y)

        logger.info(f"Created {len(X)} training sequences")
        logger.info(f"X shape: {X.shape}, y shape: {y.shape}")

        return X, y

    def train(
            self,
            df: pd.DataFrame,
            epochs: int = 50,
            batch_size: int = 32,
            lr: float = 0.0001,  # Lower default learning rate
            val_split: float = 0.2
    ):
        """
        Train the regime predictor.

        Args:
            df: DataFrame with market data
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            val_split: Validation split ratio
        """
        # Extract regimes
        self.extract_regimes(df)

        # Create training sequences
        X, y = self.create_training_sequences()

        # Check for NaN/Inf
        if np.isnan(X).any() or np.isnan(y).any():
            logger.warning("Found NaN values in data, replacing with 0")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        # Train/val split
        val_size = int(len(X) * val_split)
        X_train, X_val = X[:-val_size], X[-val_size:]
        y_train, y_val = y[:-val_size], y[-val_size:]

        # Convert to tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.FloatTensor(y_val).to(self.device)

        # Check for NaN in tensors
        if torch.isnan(X_train).any() or torch.isnan(y_train).any():
            logger.error("NaN values in training tensors after conversion!")
            X_train = torch.nan_to_num(X_train, nan=0.0)
            y_train = torch.nan_to_num(y_train, nan=0.0)

        # Initialize model
        self.model = RegimePredictor(
            input_dim=self.feature_dim,
            hidden_dims=[128, 64, 32],
            output_dim=self.feature_dim,
            dropout=0.3
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        criterion = nn.MSELoss()

        logger.info(f"Training regime predictor on {self.device}...")

        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()

            # Mini-batch training
            perm = torch.randperm(len(X_train))
            train_loss = 0.0
            n_batches = 0

            for i in range(0, len(X_train), batch_size):
                indices = perm[i:i + batch_size]
                batch_X = X_train[indices]
                batch_y = y_train[indices]

                optimizer.zero_grad()
                pred = self.model(batch_X)

                # For sequence input, take the last prediction
                if pred.dim() == 3:
                    pred = pred[:, -1, :]

                # Check for NaN in predictions
                if torch.isnan(pred).any():
                    logger.warning(f"NaN in predictions at epoch {epoch + 1}, batch {i // batch_size}")
                    continue

                loss = criterion(pred, batch_y)

                # Check for NaN loss
                if torch.isnan(loss):
                    logger.warning(f"NaN loss at epoch {epoch + 1}, skipping batch")
                    continue

                loss.backward()

                # Gradient clipping to prevent explosion
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()

                train_loss += loss.item()
                n_batches += 1

            if n_batches == 0:
                logger.error("No valid batches in epoch, stopping training")
                break

            train_loss /= n_batches

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_val)
                if val_pred.dim() == 3:
                    val_pred = val_pred[:, -1, :]

                # Replace NaN predictions with 0
                if torch.isnan(val_pred).any():
                    val_pred = torch.nan_to_num(val_pred, nan=0.0)

                val_loss = criterion(val_pred, y_val).item()

                # Check for NaN val loss
                if np.isnan(val_loss):
                    logger.warning(f"NaN validation loss at epoch {epoch + 1}")
                    val_loss = float('inf')

            scheduler.step()

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

            # Early stopping
            if val_loss < best_val_loss and not np.isnan(val_loss):
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        logger.info(f"Training complete. Best val loss: {best_val_loss:.6f}")

    def predict_future_regime(self, recent_regimes: List[Dict] = None) -> np.ndarray:
        """
        Predict the next regime features.

        Args:
            recent_regimes: Recent regime windows (uses last N if None)

        Returns:
            Predicted regime features
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train first.")

        if recent_regimes is None:
            if not self.regimes or len(self.regimes) < self.history_length:
                raise ValueError("Insufficient regimes for prediction")
            recent_regimes = self.regimes[-self.history_length:]

        # Transform regimes
        transformed = [self.regime_extractor.transform_regime(r) for r in recent_regimes]
        X = np.stack(transformed)
        X = torch.FloatTensor(X).unsqueeze(0).to(self.device)  # [1, history_length, feature_dim]

        self.model.eval()
        with torch.no_grad():
            pred = self.model(X)
            if pred.dim() == 3:
                pred = pred[:, -1, :]  # Take last prediction
            pred = pred.cpu().numpy()[0]

        return pred

    def save(self, filepath: str):
        """Save the predictor to disk."""
        # Create directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        state = {
            'model_state': self.model.state_dict() if self.model else None,
            'regime_extractor': self.regime_extractor,
            'regimes': self.regimes,
            'feature_dim': self.feature_dim,
            'config': {
                'regime_duration': self.regime_duration,
                'history_length': self.history_length
            }
        }

        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

        logger.info(f"Predictor saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'DDGDADistributionPredictor':
        """Load predictor from disk."""
        import pickle

        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        predictor = cls(
            regime_window_size=state['config'].get('regime_window_size', 20),
            regime_duration=state['config'].get('regime_duration', 60),
            history_length=state['config'].get('history_length', 5)
        )

        predictor.regime_extractor = state['regime_extractor']
        predictor.regimes = state['regimes']
        predictor.feature_dim = state['feature_dim']

        if state['model_state'] is not None:
            predictor.model = RegimePredictor(
                input_dim=predictor.feature_dim,
                hidden_dims=[128, 64, 32],
                output_dim=predictor.feature_dim,
                dropout=0.3
            ).to(predictor.device)
            predictor.model.load_state_dict(state['model_state'])

        logger.info(f"Predictor loaded from {filepath}")
        return predictor


if __name__ == "__main__":
    # Example usage
    import pandas as pd

    # Create synthetic data
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
    tickers = ['AAPL', 'MSFT']

    data_list = []
    for ticker in tickers:
        n = len(dates)
        close = 100 + np.cumsum(np.random.randn(n))

        df = pd.DataFrame({
            'Date': dates,
            'Ticker': ticker,
            'Close': close,
            'Volume': np.random.randint(1e6, 1e8, n),
            'sentiment_score': np.random.uniform(-0.5, 0.5, n),
            'rsi': np.random.uniform(30, 70, n),
            'volatility_10d': np.random.uniform(0.01, 0.03, n)
        })
        data_list.append(df)

    data = pd.concat(data_list, ignore_index=True)

    # Train predictor
    predictor = DDGDADistributionPredictor()
    predictor.train(data, epochs=30, batch_size=16)

    # Predict future regime
    future_regime = predictor.predict_future_regime()
    print(f"Predicted future regime shape: {future_regime.shape}")
