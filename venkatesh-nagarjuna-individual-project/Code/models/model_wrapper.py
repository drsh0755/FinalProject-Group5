"""
Model wrapper for easy inference and deployment.
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging

from models.tft import TFTClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TFTModelWrapper:
    """Wrapper for TFT model for easy inference."""

    def __init__(self, model: TFTClassifier, feature_cols: List[str], ticker_to_id: Dict[str, int]):
        """
        Initialize wrapper.

        Args:
            model: Trained TFT model
            feature_cols: List of feature column names
            ticker_to_id: Mapping from ticker to ID
        """
        self.model = model
        self.feature_cols = feature_cols
        self.ticker_to_id = ticker_to_id
        self.device = next(model.parameters()).device
        self.sequence_length = 60  # Default sequence length

    def save(self, save_dir: str):
        """Save model wrapper."""
        os.makedirs(save_dir, exist_ok=True)

        # Save model state
        torch.save(self.model.state_dict(), os.path.join(save_dir, 'model.pt'))

        # Save config
        config = {
            'num_features': self.model.num_features,
            'hidden_dim': self.model.hidden_dim,
            'num_heads': self.model.num_heads,
            'lstm_layers': self.model.lstm_layers,
            'num_tickers': self.model.num_tickers,
            'num_classes': self.model.num_classes,
            'dropout': self.model.dropout,
            'ticker_embed_dim': self.model.ticker_embed_dim,
            'feature_cols': self.feature_cols,
            'ticker_to_id': self.ticker_to_id
        }

        with open(os.path.join(save_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Model wrapper saved to {save_dir}")

    @classmethod
    def load(cls, load_dir: str, device: str = None) -> 'TFTModelWrapper':
        """Load model wrapper."""
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load config
        with open(os.path.join(load_dir, 'config.json'), 'r') as f:
            config = json.load(f)

        # Create model
        model = TFTClassifier(
            num_features=config['num_features'],
            hidden_dim=config['hidden_dim'],
            num_heads=config['num_heads'],
            lstm_layers=config['lstm_layers'],
            num_tickers=config['num_tickers'],
            num_classes=config.get('num_classes', 2),
            dropout=config['dropout'],
            ticker_embed_dim=config.get('ticker_embed_dim', 8)
        )

        # Load weights
        model.load_state_dict(torch.load(os.path.join(load_dir, 'model.pt'), map_location=device))
        model = model.to(device)
        model.eval()

        # Create wrapper
        wrapper = cls(
            model=model,
            feature_cols=config['feature_cols'],
            ticker_to_id=config['ticker_to_id']
        )

        logger.info(f"Model wrapper loaded from {load_dir}")
        return wrapper

    @torch.no_grad()
    def predict(self, features: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        """Make predictions for a single batch."""
        self.model.eval()

        # Move to device
        features = {k: v.to(self.device) for k, v in features.items()}

        # Forward pass
        outputs = self.model(features)
        logits = outputs['logits']

        # Get predictions
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(logits, dim=-1)

        return {
            'predictions': preds.cpu().numpy(),
            'probabilities': probs.cpu().numpy(),
            'logits': logits.cpu().numpy()
        }

    @torch.no_grad()
    def batch_predict_directions(
        self,
        df: pd.DataFrame,
        batch_size: int = 32
    ) -> pd.DataFrame:
        """
        Generate predictions for all sequences in a DataFrame.

        Args:
            df: DataFrame with Date, Ticker, and feature columns
            batch_size: Batch size for prediction

        Returns:
            DataFrame with predictions, probabilities, and dates
        """
        self.model.eval()

        predictions = []

        # Process each ticker
        for ticker in df['Ticker'].unique():
            ticker_data = df[df['Ticker'] == ticker].sort_values('Date').reset_index(drop=True)

            if len(ticker_data) < self.sequence_length:
                logger.warning(f"Skipping {ticker}: insufficient data ({len(ticker_data)} < {self.sequence_length})")
                continue

            # Create sequences
            for i in range(len(ticker_data) - self.sequence_length + 1):
                seq = ticker_data.iloc[i:i + self.sequence_length]

                # Prepare features
                missing_features = [f for f in self.feature_cols if f not in seq.columns]
                if missing_features:
                    for feat in missing_features:
                        seq[feat] = 0.0

                # Extract features
                features = seq[self.feature_cols].values
                features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

                # Get ticker ID
                ticker_id = self.ticker_to_id.get(ticker, 0)

                predictions.append({
                    'ticker': ticker,
                    'date': seq['Date'].iloc[-1],
                    'features': features,
                    'ticker_id': ticker_id
                })

        if not predictions:
            logger.warning("No valid sequences found")
            return pd.DataFrame()

        # Batch predict
        results = []

        for i in range(0, len(predictions), batch_size):
            batch = predictions[i:i + batch_size]

            # Stack features
            features_batch = torch.FloatTensor(np.stack([p['features'] for p in batch]))
            ticker_ids_batch = torch.LongTensor([[p['ticker_id']] for p in batch])

            # Move to device
            input_features = {
                'encoder_cont': features_batch.to(self.device),
                'ticker_id': ticker_ids_batch.to(self.device)
            }

            # Predict
            outputs = self.model(input_features)
            logits = outputs['logits']
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

            # Store results
            for j, pred_data in enumerate(batch):
                results.append({
                    'ticker': pred_data['ticker'],
                    'date': pred_data['date'],
                    'prediction': int(preds[j].cpu().item()),
                    'prob_down': float(probs[j, 0].cpu().item()),
                    'prob_up': float(probs[j, 1].cpu().item()),
                    'confidence': float(probs[j, preds[j]].cpu().item())
                })

        results_df = pd.DataFrame(results)
        results_df['direction'] = results_df['prediction'].map({0: 'DOWN', 1: 'UP'})

        logger.info(f"Generated {len(results_df)} predictions")

        return results_df

    def predict_direction(
        self,
        ticker: str,
        features_df: pd.DataFrame
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict direction for a single ticker.

        Args:
            ticker: Ticker symbol
            features_df: DataFrame with features (last 60 rows will be used)

        Returns:
            Tuple of (direction, confidence, probabilities)
        """
        # Take last sequence_length rows
        features_df = features_df.tail(self.sequence_length)

        if len(features_df) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} rows, got {len(features_df)}")

        # Ensure all required features exist
        missing_features = [f for f in self.feature_cols if f not in features_df.columns]
        for feat in missing_features:
            features_df[feat] = 0.0

        # Prepare features
        features = features_df[self.feature_cols].values
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        # Get ticker ID
        ticker_id = self.ticker_to_id.get(ticker, 0)

        # Create tensors
        features_tensor = torch.FloatTensor(features).unsqueeze(0)
        ticker_id_tensor = torch.LongTensor([[ticker_id]])

        input_features = {
            'encoder_cont': features_tensor,
            'ticker_id': ticker_id_tensor
        }

        # Predict
        output = self.predict(input_features)

        pred_class = int(output['predictions'][0])
        probs = output['probabilities'][0]

        direction = 'UP' if pred_class == 1 else 'DOWN'
        confidence = float(probs[pred_class])
        probabilities = {
            'up': float(probs[1]),
            'down': float(probs[0])
        }

        return direction, confidence, probabilities
