"""
Temporal Fusion Transformer (TFT) implemented in PyTorch.
Optimized for stock direction forecasting with GPU acceleration.

Based on: "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"
https://arxiv.org/abs/1912.09363
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GatedLinearUnit(nn.Module):
    """Gated Linear Unit (GLU) for context-aware feature gating."""

    def __init__(self, input_dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.1):
        """
        Initialize GLU.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension (defaults to input_dim)
            dropout: Dropout rate
        """
        super().__init__()

        if hidden_dim is None:
            hidden_dim = input_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [..., input_dim]

        Returns:
            Gated output [..., hidden_dim]
        """
        sig = torch.sigmoid(self.fc1(x))
        lin = self.fc2(x)
        out = sig * lin
        out = self.dropout(out)
        return out


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network (GRN) with optional context."""

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            context_dim: Optional[int] = None,
            dropout: float = 0.1
    ):
        """
        Initialize GRN.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            context_dim: Context dimension (optional)
            dropout: Dropout rate
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim

        # Primary network
        self.fc1 = nn.Linear(input_dim, hidden_dim)

        # Context network (if context is provided)
        if context_dim is not None:
            self.context_fc = nn.Linear(context_dim, hidden_dim, bias=False)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

        # GLU for gating
        self.glu = GatedLinearUnit(hidden_dim, output_dim, dropout=dropout)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(output_dim)

        # Skip connection
        if input_dim != output_dim:
            self.skip_fc = nn.Linear(input_dim, output_dim)
        else:
            self.skip_fc = None

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [..., input_dim]
            context: Optional context tensor [..., context_dim]

        Returns:
            Output tensor [..., output_dim]
        """
        # Skip connection
        if self.skip_fc is not None:
            skip = self.skip_fc(x)
        else:
            skip = x

        # Primary network
        hidden = F.elu(self.fc1(x))

        # Add context if provided
        if context is not None and self.context_dim is not None:
            hidden = hidden + self.context_fc(context)

        hidden = F.elu(self.fc2(hidden))
        hidden = self.dropout(hidden)

        # Gating
        gated = self.glu(hidden)

        # Residual connection
        out = self.layer_norm(skip + gated)

        return out


class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network for feature importance."""

    def __init__(
            self,
            input_dim: int,
            num_inputs: int,
            hidden_dim: int,
            dropout: float = 0.1,
            context_dim: Optional[int] = None
    ):
        """
        Initialize Variable Selection Network.

        Args:
            input_dim: Dimension per input variable
            num_inputs: Number of input variables
            hidden_dim: Hidden dimension
            dropout: Dropout rate
            context_dim: Context dimension (optional)
        """
        super().__init__()

        self.input_dim = input_dim
        self.num_inputs = num_inputs
        self.hidden_dim = hidden_dim

        # Flatten all inputs
        self.flattened_grn = GatedResidualNetwork(
            input_dim=num_inputs * input_dim,
            hidden_dim=hidden_dim,
            output_dim=num_inputs,
            context_dim=context_dim,
            dropout=dropout
        )

        # Per-variable processing
        self.single_variable_grns = nn.ModuleList([
            GatedResidualNetwork(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                dropout=dropout
            )
            for _ in range(num_inputs)
        ])

    def forward(
            self,
            variables: torch.Tensor,
            context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            variables: Input variables [batch_size, num_inputs, input_dim]
            context: Optional context [batch_size, context_dim]

        Returns:
            Tuple of (weighted_sum, weights)
            - weighted_sum: [batch_size, hidden_dim]
            - weights: [batch_size, num_inputs]
        """
        batch_size = variables.size(0)

        # Flatten variables
        flattened = variables.view(batch_size, -1)

        # Compute variable selection weights
        var_weights = self.flattened_grn(flattened, context)
        var_weights = F.softmax(var_weights, dim=-1)  # [batch_size, num_inputs]

        # Process each variable
        processed_vars = []
        for i, grn in enumerate(self.single_variable_grns):
            processed = grn(variables[:, i, :])  # [batch_size, hidden_dim]
            processed_vars.append(processed)

        processed_vars = torch.stack(processed_vars, dim=1)  # [batch_size, num_inputs, hidden_dim]

        # Weight and sum
        weighted_sum = torch.sum(
            var_weights.unsqueeze(-1) * processed_vars,
            dim=1
        )  # [batch_size, hidden_dim]

        return weighted_sum, var_weights


class InterpretableMultiHeadAttention(nn.Module):
    """Multi-head attention with interpretability."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """
        Initialize multi-head attention.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            q: Query [batch_size, seq_len, d_model]
            k: Key [batch_size, seq_len, d_model]
            v: Value [batch_size, seq_len, d_model]
            mask: Optional mask [batch_size, seq_len, seq_len]

        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, _ = q.size()

        # Linear projections and reshape for multi-head
        Q = self.w_q(q).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(k).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(v).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        context = torch.matmul(attn_weights, V)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Final linear projection
        output = self.w_o(context)

        # Average attention weights across heads for interpretability
        avg_attn_weights = attn_weights.mean(dim=1)  # [batch_size, seq_len, seq_len]

        return output, avg_attn_weights


class TemporalFusionTransformer(nn.Module):
    """
    Complete Temporal Fusion Transformer for stock direction forecasting.
    Optimized for binary classification (up/down).
    """

    def __init__(
            self,
            # Feature dimensions
            num_continuous_features: int,
            num_categorical_features: int = 0,
            categorical_embedding_dims: Optional[List[int]] = None,
            categorical_cardinalities: Optional[List[int]] = None,

            # Static features
            num_static_continuous: int = 0,
            num_static_categorical: int = 0,
            static_categorical_cardinalities: Optional[List[int]] = None,

            # Architecture
            hidden_dim: int = 128,
            lstm_layers: int = 2,
            num_heads: int = 4,
            dropout: float = 0.1,

            # Output
            num_classes: int = 2,

            # Sequence
            sequence_length: int = 60
    ):
        """
        Initialize TFT.

        Args:
            num_continuous_features: Number of continuous time-varying features
            num_categorical_features: Number of categorical time-varying features
            categorical_embedding_dims: Embedding dimensions for categorical features
            categorical_cardinalities: Cardinalities for categorical features
            num_static_continuous: Number of static continuous features
            num_static_categorical: Number of static categorical features
            static_categorical_cardinalities: Cardinalities for static categorical features
            hidden_dim: Hidden dimension throughout the network
            lstm_layers: Number of LSTM layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            num_classes: Number of output classes
            sequence_length: Input sequence length
        """
        super().__init__()

        self.num_continuous_features = num_continuous_features
        self.num_categorical_features = num_categorical_features
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.num_classes = num_classes

        # Categorical embeddings
        self.categorical_embeddings = nn.ModuleList()
        if num_categorical_features > 0 and categorical_cardinalities:
            if categorical_embedding_dims is None:
                categorical_embedding_dims = [min(50, (card + 1) // 2) for card in categorical_cardinalities]

            for card, emb_dim in zip(categorical_cardinalities, categorical_embedding_dims):
                self.categorical_embeddings.append(nn.Embedding(card, emb_dim))

        # Static categorical embeddings
        self.static_categorical_embeddings = nn.ModuleList()
        if num_static_categorical > 0 and static_categorical_cardinalities:
            for card in static_categorical_cardinalities:
                emb_dim = min(50, (card + 1) // 2)
                self.static_categorical_embeddings.append(nn.Embedding(card, emb_dim))

        # Static covariate encoders
        total_static_dim = num_static_continuous
        if static_categorical_cardinalities:
            total_static_dim += sum([min(50, (card + 1) // 2) for card in static_categorical_cardinalities])

        self.static_context_grn = None
        if total_static_dim > 0:
            self.static_context_grn = GatedResidualNetwork(
                input_dim=total_static_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                dropout=dropout
            )

        # Variable selection for time-varying features
        total_time_varying_dim = num_continuous_features
        if categorical_embedding_dims:
            total_time_varying_dim += sum(categorical_embedding_dims)

        self.input_projection = nn.Linear(num_continuous_features, hidden_dim)

        self.temporal_vsn = VariableSelectionNetwork(
            input_dim=hidden_dim,
            num_inputs=1,  # Simplified: treating all features as one group
            hidden_dim=hidden_dim,
            dropout=dropout,
            context_dim=hidden_dim if total_static_dim > 0 else None
        )

        # LSTM encoder/decoder
        self.lstm_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0,
            batch_first=True
        )

        # Gated skip connection over LSTM
        self.post_lstm_grn = GatedResidualNetwork(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            dropout=dropout
        )

        # Static enrichment
        self.static_enrichment = None
        if total_static_dim > 0:
            self.static_enrichment = GatedResidualNetwork(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                context_dim=hidden_dim,
                dropout=dropout
            )

        # Multi-head attention
        self.attention = InterpretableMultiHeadAttention(
            d_model=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # Post-attention GRN
        self.post_attention_grn = GatedResidualNetwork(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            dropout=dropout
        )

        # Position-wise feed-forward
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Output layers
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # Initialize weights
        self.apply(self._init_weights)

        logger.info(f"TFT initialized: hidden_dim={hidden_dim}, lstm_layers={lstm_layers}, "
                    f"num_heads={num_heads}, num_classes={num_classes}")

    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.1)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    nn.init.zeros_(param.data)

    def forward(
            self,
            continuous_features: torch.Tensor,
            categorical_features: Optional[torch.Tensor] = None,
            static_continuous: Optional[torch.Tensor] = None,
            static_categorical: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            continuous_features: Time-varying continuous features [batch_size, seq_len, num_features]
            categorical_features: Time-varying categorical features [batch_size, seq_len, num_cat_features]
            static_continuous: Static continuous features [batch_size, num_static_cont]
            static_categorical: Static categorical features [batch_size, num_static_cat]

        Returns:
            Dictionary containing:
                - 'logits': Class logits [batch_size, num_classes]
                - 'attention_weights': Attention weights [batch_size, seq_len, seq_len]
                - 'variable_selection_weights': Feature importance (if available)
        """
        batch_size, seq_len, _ = continuous_features.size()

        # Process static features
        static_context = None
        if self.static_context_grn is not None:
            static_features = []

            if static_continuous is not None and static_continuous.numel() > 0:
                static_features.append(static_continuous)

            if static_categorical is not None and len(self.static_categorical_embeddings) > 0:
                for i, emb in enumerate(self.static_categorical_embeddings):
                    static_features.append(emb(static_categorical[:, i]))

            if static_features:
                static_input = torch.cat(static_features, dim=-1)
                static_context = self.static_context_grn(static_input)

        # Project continuous features
        temporal_input = self.input_projection(continuous_features)

        # Variable selection
        temporal_input_reshaped = temporal_input.view(batch_size * seq_len, 1, self.hidden_dim)

        if static_context is not None:
            static_context_expanded = static_context.unsqueeze(1).repeat(1, seq_len, 1)
            static_context_expanded = static_context_expanded.view(batch_size * seq_len, self.hidden_dim)
        else:
            static_context_expanded = None

        selected_temporal, vsn_weights = self.temporal_vsn(
            temporal_input_reshaped,
            static_context_expanded
        )

        selected_temporal = selected_temporal.view(batch_size, seq_len, self.hidden_dim)

        # LSTM processing
        lstm_out, (h_n, c_n) = self.lstm_encoder(selected_temporal)

        # Gated residual connection around LSTM
        lstm_out = self.post_lstm_grn(lstm_out)

        # Static enrichment
        if self.static_enrichment is not None and static_context is not None:
            static_context_expanded = static_context.unsqueeze(1).expand(-1, seq_len, -1)
            lstm_out = self.static_enrichment(lstm_out, static_context_expanded)

        # Self-attention
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)

        # Residual connection
        attn_out = lstm_out + attn_out

        # Post-attention processing
        enriched = self.post_attention_grn(attn_out)

        # Feed-forward
        ff_out = self.feed_forward(enriched)
        output = self.layer_norm(enriched + ff_out)

        # Use last time step for classification
        final_output = output[:, -1, :]  # [batch_size, hidden_dim]

        # Classification head
        logits = self.output_layer(final_output)  # [batch_size, num_classes]

        return {
            'logits': logits,
            'attention_weights': attn_weights,
            'variable_selection_weights': vsn_weights.view(batch_size, seq_len, -1).mean(dim=1)
        }


class TFTClassifier(nn.Module):
    """Temporal Fusion Transformer for stock direction classification."""

    def __init__(
            self,
            num_features: int,
            hidden_dim: int = 128,
            num_heads: int = 4,
            lstm_layers: int = 2,  # Changed from num_layers
            num_tickers: int = 5,
            num_classes: int = 2,
            dropout: float = 0.1,
            ticker_embed_dim: int = 8
    ):
        super().__init__()

        # Store config
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.lstm_layers = lstm_layers  # Changed from num_layers
        self.num_tickers = num_tickers
        self.num_classes = num_classes
        self.dropout = dropout
        self.ticker_embed_dim = ticker_embed_dim

        # Ticker embedding
        self.ticker_embedding = nn.Embedding(num_tickers, ticker_embed_dim)

        # Input projection with layer norm
        self.input_projection = nn.Sequential(
            nn.Linear(num_features + ticker_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # LSTM encoder
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=lstm_layers,  # Use lstm_layers
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,  # Use lstm_layers
            bidirectional=False
        )

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer norms
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # Initialize weights
        self.apply(self._init_weights)

        logger.info(
            f"TFT initialized: hidden_dim={hidden_dim}, lstm_layers={lstm_layers}, "
            f"num_heads={num_heads}, num_classes={num_classes}"
        )

    def _init_weights(self, module):
        """Initialize weights with smaller values."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.1)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.01)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param, gain=0.1)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        encoder_cont = features['encoder_cont']
        ticker_id = features['ticker_id']

        batch_size, seq_len, _ = encoder_cont.shape

        # Input validation and clipping
        encoder_cont = torch.nan_to_num(encoder_cont, nan=0.0, posinf=1.0, neginf=-1.0)
        encoder_cont = torch.clamp(encoder_cont, -10, 10)

        # Ticker embedding
        ticker_emb = self.ticker_embedding(ticker_id.squeeze(-1))
        ticker_emb = ticker_emb.unsqueeze(1).expand(-1, seq_len, -1)

        # Concatenate and project
        x = torch.cat([encoder_cont, ticker_emb], dim=-1)
        x = self.input_projection(x)

        # LSTM
        lstm_out, _ = self.lstm(x)

        # Attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Residual + norm
        x = self.ln1(lstm_out + attn_out)

        # Take last timestep
        x = x[:, -1, :]

        # Classify
        logits = self.classifier(x)
        logits = torch.clamp(logits, -10, 10)

        return {'logits': logits, 'hidden': x}


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test TFT
    logger.info("Testing Temporal Fusion Transformer...")

    batch_size = 16
    seq_len = 60
    num_features = 10
    num_tickers = 5

    # Create sample data
    features_dict = {
        'encoder_cont': torch.randn(batch_size, seq_len, num_features),
        'ticker_id': torch.randint(0, num_tickers, (batch_size, 1))
    }

    # Initialize model
    model = TFTClassifier(
        num_features=num_features,
        num_tickers=num_tickers,
        hidden_dim=128,
        lstm_layers=2,
        num_heads=4,
        dropout=0.1
    )

    logger.info(f"Model parameters: {count_parameters(model):,}")

    # Forward pass
    outputs = model(features_dict)

    logger.info(f"Logits shape: {outputs['logits'].shape}")
    logger.info(f"Attention weights shape: {outputs['attention_weights'].shape}")
    logger.info(f"Variable selection weights shape: {outputs['variable_selection_weights'].shape}")

    # Test with CUDA if available
    if torch.cuda.is_available():
        logger.info("\nTesting on GPU...")
        device = torch.device('cuda')
        model = model.to(device)

        features_dict_gpu = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in features_dict.items()
        }

        with torch.cuda.amp.autocast():
            outputs = model(features_dict_gpu)

        logger.info("GPU forward pass successful!")
        logger.info(f"Output device: {outputs['logits'].device}")

    logger.info("\n✓ TFT test complete!")
