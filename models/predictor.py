"""
Transformer-based action-conditioned predictor that interleaves autoencoder features with actions.
"""

import torch
import torch.nn as nn

from models.encoder_layer_with_attn import EncoderLayerWithAttn


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
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.max_sequence_length = max_sequence_length
        self.level = level
        self.max_lookahead = 10  # For compatibility with old interface

        # Action embedding layer
        self.action_embedding = nn.Linear(action_dim, embed_dim)

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

        # Output head to predict next encoder features
        self.output_head = nn.Linear(embed_dim, embed_dim)

        # Layer norm
        self.layer_norm = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def encode_action(self, action_dict, device):
        """Convert action dictionary to embedding vector."""
        action_vector = torch.zeros(self.action_dim, device=device)

        if 'motor_left' in action_dict:
            action_vector[0] = action_dict['motor_left']
        if 'motor_right' in action_dict:
            action_vector[1] = action_dict['motor_right']
        if 'duration' in action_dict:
            action_vector[2] = action_dict['duration']

        return action_vector

    @staticmethod
    def create_causal_mask(seq_len, device):
        """Create causal attention mask to prevent attending to future tokens."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, encoder_features_history, actions, return_attn=False):
        """
        Forward pass with interleaved encoder features and actions.

        Args:
            encoder_features_history: list[Tensor] of shape [B, num_patches+1, D]
            actions: list of action dicts (length = len(encoder_features_history) - 1)
            return_attn: whether to record and return per-layer attention maps

        Returns:
            predicted_features: [B, num_patches+1, D]
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
                action_vec = self.encode_action(actions[i], device=device)
                action_vectors = action_vec.expand(batch_size, -1)
                action_embeds = self.action_embedding(action_vectors)
                sequence.append(action_embeds)
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

        # Run transformer layers, optionally recording attention
        attn_records = [] if return_attn else None
        for layer in self.transformer_layers:
            x, attn = layer(x, attn_mask=causal_mask, record_attn=return_attn)
            if return_attn:
                attn_records.append(attn)

        x = self.layer_norm(x)

        # Get predictions from the future query slots (the final N positions)
        pred_tokens = x[:, -num_patches_plus_one:, :]
        predicted_features = self.output_head(pred_tokens)

        if not return_attn:
            return predicted_features

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

        return predicted_features, attn_info

    def predict_uncertainty(self, encoder_features_history, actions, num_samples=10):
        """
        Predict uncertainty using dropout at inference time (Monte Carlo dropout).

        Returns:
            mean_prediction: [batch_size, num_patches+1, embed_dim]
            uncertainty: [batch_size] - variance across samples
        """
        self.train()  # Enable dropout

        predictions = []
        for _ in range(num_samples):
            with torch.no_grad():
                pred = self.forward(encoder_features_history, actions)
                predictions.append(pred)

        predictions = torch.stack(predictions, dim=0)  # [num_samples, batch_size, num_patches+1, embed_dim]

        mean_prediction = predictions.mean(dim=0)
        uncertainty = predictions.var(dim=0).mean(dim=-1).mean(dim=-1)  # Average variance across patch and embedding dims

        return mean_prediction, uncertainty

