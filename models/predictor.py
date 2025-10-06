"""
Transformer-based action-conditioned predictor that interleaves autoencoder features with actions.
"""

import torch
import torch.nn as nn

from models.encoder_layer_with_attn import EncoderLayerWithAttn
import config


def action_dict_to_index(action_dict, action_space):
    """
    Convert an action dictionary to its index in the action space.

    Maps a continuous action dictionary to the closest discrete action index.
    This is used for action reconstruction loss, where the predictor must
    identify which discrete action was taken from the predicted features.

    Args:
        action_dict: Dictionary with action parameters (motor_left, motor_right, duration)
                    Example: {'motor_left': 0, 'motor_right': 0.12, 'duration': 0.2}
        action_space: List of action dictionaries defining the discrete action space
                     Example: [{'motor_left': 0, 'motor_right': 0, 'duration': 0.2},
                               {'motor_right': 0, 'motor_right': 0.12, 'duration': 0.2}]

    Returns:
        int: Index of the action in action_space
             For the examples above, action_dict would return index 1
             Returns -1 if action not found in action space

    Example:
        >>> action_space = [
        ...     {'motor_left': 0, 'motor_right': 0, 'duration': 0.2},      # index 0: stop
        ...     {'motor_left': 0, 'motor_right': 0.12, 'duration': 0.2}   # index 1: forward
        ... ]
        >>> action = {'motor_left': 0, 'motor_right': 0.12, 'duration': 0.2}
        >>> action_dict_to_index(action, action_space)
        1
    """
    for idx, space_action in enumerate(action_space):
        # Check if all relevant keys match (with tolerance for floating point comparison)
        match = True
        for key in config.ACTION_CHANNELS:
            action_val = action_dict.get(key, 0.0)
            space_val = space_action.get(key, 0.0)
            if abs(action_val - space_val) > 1e-6:
                match = False
                break
        if match:
            return idx
    return -1  # Action not found in action space


class ActionEmbedding(nn.Module):
    """
    Learns a compact embedding of normalized action vectors.

    Takes normalized action channels (motor_left, motor_right, duration) in [-1, 1]
    and produces a learned embedding suitable for FiLM conditioning.
    """

    def __init__(self, in_dim, emb_dim=None, hidden_dim=None):
        """
        Args:
            in_dim: Number of input action channels (e.g., 3 for motor_left, motor_right, duration)
            emb_dim: Output embedding dimension (default: config.ACTION_EMBED_DIM)
            hidden_dim: Hidden layer width (default: config.FILM_HIDDEN_DIM)
        """
        super().__init__()
        if emb_dim is None:
            emb_dim = config.ACTION_EMBED_DIM
        if hidden_dim is None:
            hidden_dim = config.FILM_HIDDEN_DIM

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, emb_dim),
            nn.GELU(),
            nn.LayerNorm(emb_dim),
        )

    def forward(self, action_normalized):
        """
        Args:
            action_normalized: (batch_size, in_dim) tensor in [-1, 1]

        Returns:
            action_embedding: (batch_size, emb_dim) learned action representation
        """
        return self.net(action_normalized)


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation layer.

    Produces per-channel affine transformation parameters (gamma, beta) from
    action embeddings to condition feature representations on actions.

    Applies: output = gamma * features + beta
    """

    def __init__(self, emb_dim, num_channels):
        """
        Args:
            emb_dim: Dimension of input action embedding
            num_channels: Number of feature channels to modulate
        """
        super().__init__()
        self.to_gamma = nn.Linear(emb_dim, num_channels)
        self.to_beta = nn.Linear(emb_dim, num_channels)

        # Initialize near identity: gamma ~ 1, beta ~ 0
        # This ensures FiLM starts as a near-identity transform
        nn.init.zeros_(self.to_gamma.weight)
        nn.init.zeros_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight)
        nn.init.zeros_(self.to_beta.bias)

    def forward(self, action_embedding):
        """
        Args:
            action_embedding: (batch_size, emb_dim) action representation

        Returns:
            gamma: (batch_size, num_channels) scale parameters
            beta: (batch_size, num_channels) shift parameters
        """
        gamma = 1.0 + self.to_gamma(action_embedding)  # Start near 1
        beta = self.to_beta(action_embedding)          # Start near 0
        return gamma, beta


def film_apply(features, gamma, beta):
    """
    Apply FiLM affine transformation to features.

    Args:
        features: (batch_size, num_channels, ...) feature tensor
        gamma: (batch_size, num_channels) scale parameters
        beta: (batch_size, num_channels) shift parameters

    Returns:
        modulated_features: (batch_size, num_channels, ...) with same shape as features
    """
    if features.ndim < 2:
        raise ValueError("features tensor must have at least 2 dimensions")
    if gamma.shape[0] != features.shape[0] or beta.shape[0] != features.shape[0]:
        raise ValueError("gamma and beta batch sizes must match features")
    if gamma.shape[-1] != features.shape[-1] or beta.shape[-1] != features.shape[-1]:
        raise ValueError("gamma and beta channel sizes must match features")

    broadcast_shape = [gamma.shape[0]] + [1] * (features.ndim - 2) + [gamma.shape[-1]]
    gamma = gamma.view(*broadcast_shape)
    beta = beta.view(*broadcast_shape)

    return features * gamma + beta


class TransformerActionConditionedPredictor(nn.Module):
    """
    Transformer-based predictor that interleaves autoencoder encoder features with actions
    using causal attention masking for future state prediction.
    """

    def __init__(
        self,
        embed_dim=256,
        action_dim=32,
        num_heads=8,
        num_layers=6,
        max_sequence_length=4096,
        dropout=0.1,
        level=0,
        num_actions=2,  # Number of discrete actions in the action space
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.max_sequence_length = max_sequence_length
        self.level = level
        self.max_lookahead = 10  # For compatibility with old interface
        self.num_actions = num_actions

        # New FiLM-based action conditioning
        # ActionEmbedding: converts normalized actions to learned embeddings
        num_action_channels = len(config.ACTION_CHANNELS)
        self.action_embed = ActionEmbedding(
            in_dim=num_action_channels,
            emb_dim=config.ACTION_EMBED_DIM,
            hidden_dim=config.FILM_HIDDEN_DIM
        )

        # FiLM layers: one per transformer layer that we want to modulate
        self.film_layers = nn.ModuleDict()
        for layer_id in config.FILM_BLOCK_IDS:
            if layer_id < num_layers:
                self.film_layers[str(layer_id)] = FiLM(
                    emb_dim=config.ACTION_EMBED_DIM,
                    num_channels=embed_dim  # Transformer uses embed_dim as channel dimension
                )

        # Projection to map action embeddings to transformer token dimension
        self.action_token_proj = nn.Linear(config.ACTION_EMBED_DIM, embed_dim)
        # Positional embeddings for the full sequence
        self.position_embedding = nn.Embedding(max_sequence_length, embed_dim)

        # Token type embeddings (to distinguish encoder features from action tokens)
        self.token_type_embedding = nn.Embedding(3, embed_dim)  # 0: encoder features, 1: action, 2: future query

        # Future query slots for predicting next frame
        self.future_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 1e-2)

        # Transformer layers with causal attention
        self.transformer_layers = nn.ModuleList(
            [
                EncoderLayerWithAttn(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dropout=dropout,
                    dim_feedforward=4,
                )
                for _ in range(num_layers)
            ]
        )

        # Output head to predict next encoder features (or delta features)
        self.output_head = nn.Linear(embed_dim, embed_dim)

        # Action reconstruction head - recovers which action was taken from predicted features
        self.action_classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

        # Layer norm
        self.layer_norm = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def _normalize_action(self, action_dict, device):
        values = []
        for key in config.ACTION_CHANNELS:
            value = float(action_dict.get(key, 0.0))
            min_val, max_val = config.ACTION_RANGES[key]
            value = max(min(value, max_val), min_val)
            if max_val == min_val:
                scaled = 0.0
            else:
                scaled = 2.0 * (value - min_val) / (max_val - min_val) - 1.0
            values.append(scaled)
        return torch.tensor(values, device=device, dtype=torch.float32).unsqueeze(0)

    @staticmethod
    def create_causal_mask(seq_len, device):
        """Create causal attention mask to prevent attending to future tokens."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, encoder_features_history, actions, return_attn=False, action_normalized=None, last_features=None, return_action_logits=False):
        """
        Forward pass with interleaved encoder features and actions.

        Args:
            encoder_features_history: list[Tensor] of shape [B, num_patches+1, D]
            actions: list of action dicts (length = len(encoder_features_history) - 1)
            return_attn: whether to record and return per-layer attention maps
            action_normalized: (batch_size, num_action_channels) tensor in [-1, 1]
                             If provided, uses FiLM conditioning instead of legacy action embedding
            last_features: (batch_size, num_patches+1, D) features from current frame for delta prediction
            return_action_logits: whether to return action classification logits

        Returns:
            predicted_features: [B, num_patches+1, D] (absolute or delta, depending on config.DELTA_LATENT)
            action_logits (optional): [B, num_actions] when return_action_logits=True
            attn_info (optional): dict with attention maps and token metadata when return_attn=True
        """
        # Handle empty history case by returning random features
        if not encoder_features_history:
            batch_size = 1
            num_patches = 196  # Default for 224x224 image with 16x16 patches
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            random_features = torch.randn(batch_size, num_patches + 1, self.embed_dim, device=device)
            if return_attn:
                return random_features, None
            return random_features

        batch_size = encoder_features_history[0].shape[0]
        device = encoder_features_history[0].device

        # Compute action embedding once for FiLM conditioning (if using new path)
        film_action_embedding = None
        if action_normalized is not None:
            if action_normalized.shape[0] != batch_size:
                action_normalized = action_normalized.expand(batch_size, -1)
            film_action_embedding = self.action_embed(action_normalized)  # (batch_size, ACTION_EMBED_DIM)

        # Create interleaved sequence: FEATURES0, ACT0, FEATURES1, ACT1, ..., FEATURESn
        sequence = []
        token_types = []

        for i, features in enumerate(encoder_features_history):
            num_patches_plus_one = features.shape[1]
            # Add encoder features (CLS + patches)
            sequence.extend([features[:, j, :] for j in range(num_patches_plus_one)])
            token_types.extend([0] * num_patches_plus_one)

            # Add action token (except for the last frame)
            if i < len(actions):
                action_norm = self._normalize_action(actions[i], device=device).expand(batch_size, -1)
                action_embedding = self.action_embed(action_norm)
                action_token = self.action_token_proj(action_embedding)
                sequence.append(action_token)
                token_types.append(1)

        # Add future query slots after the last action
        future_slots = self.future_query.expand(batch_size, num_patches_plus_one, -1)
        for j in range(num_patches_plus_one):
            sequence.append(future_slots[:, j, :])
            token_types.append(2)

        # Stack into sequence tensor
        sequence_tensor = torch.stack(sequence, dim=1)  # [B, seq_len, D]
        seq_len = sequence_tensor.shape[1]

        # Guard: if the sequence is longer than the pos-embedding table, keep only the most recent tokens
        if seq_len > self.max_sequence_length:
            start = seq_len - self.max_sequence_length
            sequence_tensor = sequence_tensor[:, start:, :]
            token_types = token_types[start:]
            seq_len = sequence_tensor.shape[1]

        # Add positional embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_embeds = self.position_embedding(positions)

        # Add token type embeddings
        token_type_tensor = torch.tensor(token_types, device=device, dtype=torch.long)
        type_embeds = self.token_type_embedding(token_type_tensor.unsqueeze(0).expand(batch_size, -1))

        # Combine embeddings
        x = sequence_tensor + pos_embeds + type_embeds
        x = self.dropout(x)

        # Create causal attention mask
        causal_mask = self.create_causal_mask(seq_len, device=device)

        # Run transformer layers with FiLM conditioning, optionally recording attention
        attn_records = [] if return_attn else None
        for layer_idx, layer in enumerate(self.transformer_layers):
            # Apply transformer layer
            x, attn = layer(x, attn_mask=causal_mask, record_attn=return_attn)
            if return_attn:
                attn_records.append(attn)

            # Apply FiLM conditioning after specified layers
            if film_action_embedding is not None and str(layer_idx) in self.film_layers:
                gamma, beta = self.film_layers[str(layer_idx)](film_action_embedding)
                # x has shape (batch_size, seq_len, embed_dim)
                # gamma, beta have shape (batch_size, embed_dim)
                x = film_apply(x, gamma, beta)

        x = self.layer_norm(x)

        # Get predictions from the future query slots (the final N positions)
        pred_tokens = x[:, -num_patches_plus_one:, :]
        predicted_features = self.output_head(pred_tokens)

        # Apply delta latent prediction if enabled
        if config.DELTA_LATENT and last_features is not None:
            # predicted_features is delta, add to last_features to get absolute prediction
            predicted_features = last_features + predicted_features

        # Compute action classification logits from future query tokens
        action_logits = None
        if return_action_logits:
            # Get the future query predictions (last N tokens)
            num_future_tokens = num_patches_plus_one
            future_predictions = predicted_features[:, -num_future_tokens:, :]

            # Pool across future predictions
            action_logits = self.action_classifier(future_predictions.mean(dim=1))

        if not return_attn and not return_action_logits:
            return predicted_features

        if return_action_logits and not return_attn:
            return predicted_features, action_logits

        # Derive token indices for downstream analysis
        token_types_t = token_type_tensor
        is_frame = token_types_t == 0
        is_action = token_types_t == 1
        is_future = token_types_t == 2

        future_idx = torch.nonzero(is_future, as_tuple=True)[0]
        action_idx = torch.nonzero(is_action, as_tuple=True)[0]
        frame_idx = torch.nonzero(is_frame, as_tuple=True)[0]

        last_action_pos = action_idx[-1].item() if action_idx.numel() > 0 else None

        last_frame_mask = torch.zeros_like(token_types_t, dtype=torch.bool)
        if last_action_pos is not None:
            j = last_action_pos - 1
            while j >= 0 and is_frame[j]:
                last_frame_mask[j] = True
                j -= 1
        last_frame_idx = torch.nonzero(last_frame_mask, as_tuple=True)[0]

        attn_info = {
            'attn': attn_records,
            'token_types': token_types_t,
            'future_idx': future_idx,
            'action_idx': action_idx,
            'frame_idx': frame_idx,
            'last_action_pos': last_action_pos,
            'last_frame_idx': last_frame_idx,
        }

        if return_action_logits:
            return predicted_features, action_logits, attn_info
        return predicted_features, attn_info

    def predict_uncertainty(self, encoder_features_history, actions, num_samples=10):
        """
        Predict uncertainty using dropout at inference time (Monte Carlo dropout).

        Returns:
            mean_prediction: [batch_size, num_patches+1, embed_dim]
            uncertainty: [batch_size] - variance across samples
        """
        self.train()  # Enable dropout

        device = (
            encoder_features_history[0].device
            if encoder_features_history
            else self.future_query.device
        )
        action_normalized = None
        if actions:
            action_normalized = self._normalize_action(actions[-1], device=device)
        last_features = encoder_features_history[-1].detach() if encoder_features_history else None

        predictions = []
        for _ in range(num_samples):
            with torch.no_grad():
                pred = self.forward(
                    encoder_features_history,
                    actions,
                    action_normalized=action_normalized,
                    last_features=last_features,
                )
                predictions.append(pred)

        predictions = torch.stack(predictions, dim=0)  # [num_samples, batch_size, num_patches+1, embed_dim]

        mean_prediction = predictions.mean(dim=0)
        uncertainty = predictions.var(dim=0).mean(dim=-1).mean(dim=-1)  # Average variance across patch and embedding dims

        return mean_prediction, uncertainty

