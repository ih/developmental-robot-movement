"""
Transformer-based action-conditioned predictor that interleaves autoencoder features with actions.
"""

import torch
import torch.nn as nn


class TransformerActionConditionedPredictor(nn.Module):
    """
    Transformer-based predictor that interleaves autoencoder encoder features with actions
    using causal attention masking for future state prediction.
    """
    def __init__(self, embed_dim=256, action_dim=32, num_heads=8, num_layers=6, 
                 max_sequence_length=4096, dropout=0.1, level=0):
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
        self.token_type_embedding = nn.Embedding(3, embed_dim)  # 0: encoder features, 1: Action, 2: future query
        
        # Future query slots for predicting next frame
        self.future_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 1e-2)
        
        # Transformer layers with causal attention
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=4 * embed_dim,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Output head to predict next encoder features
        self.output_head = nn.Linear(embed_dim, embed_dim)
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def encode_action(self, action_dict):
        """Convert action dictionary to embedding vector"""
        # Create a fixed-size vector from action dictionary
        action_vector = torch.zeros(self.action_dim)
        
        if 'motor_left' in action_dict:
            action_vector[0] = action_dict['motor_left']
        if 'motor_right' in action_dict:
            action_vector[1] = action_dict['motor_right'] 
        if 'duration' in action_dict:
            action_vector[2] = action_dict['duration']
        
        return action_vector
    
    def create_causal_mask(self, seq_len):
        """Create causal attention mask to prevent attending to future tokens"""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, encoder_features_history, actions):
        """
        Forward pass with interleaved encoder features and actions
        
        Args:
            encoder_features_history: List of [batch_size, num_patches+1, embed_dim] - Features from autoencoder forward_encoder
            actions: List of action dicts, length = len(encoder_features_history) - 1
            
        Returns:
            predicted_features: [batch_size, num_patches+1, embed_dim] - Predicted next encoder features
        """
        # Handle empty history case by returning random features
        if not encoder_features_history:
            # Return random features with appropriate shape
            batch_size = 1
            num_patches = 196  # Default for 224x224 image with 16x16 patches
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            return torch.randn(batch_size, num_patches + 1, self.embed_dim, device=device)
        
        batch_size = encoder_features_history[0].shape[0]
        device = encoder_features_history[0].device
        
        # Create interleaved sequence: FEATURES0, ACT0, FEATURES1, ACT1, ..., FEATURESn
        sequence = []
        token_types = []
        
        for i, features in enumerate(encoder_features_history):
            # Add all encoder features (including CLS and patch tokens)
            num_patches_plus_one = features.shape[1]
            for j in range(num_patches_plus_one):
                sequence.append(features[:, j, :])
                token_types.append(0)  # Encoder feature token type
            
            # Add action token (except for the last frame)
            if i < len(actions):
                action_vectors = torch.stack([
                    self.encode_action(actions[i]).to(device) 
                    for _ in range(batch_size)
                ])
                action_embeds = self.action_embedding(action_vectors)
                sequence.append(action_embeds)
                token_types.append(1)  # Action token type
        
        # Add future query slots after the last action
        future_slots = self.future_query.expand(batch_size, num_patches_plus_one, -1)
        for j in range(num_patches_plus_one):
            sequence.append(future_slots[:, j, :])
            token_types.append(2)  # Future query token type
        
        # Stack into sequence tensor
        sequence_tensor = torch.stack(sequence, dim=1)  # [B, seq_len, D]
        seq_len = sequence_tensor.shape[1]

        # Guard: if the sequence is longer than the pos-embedding table,
        # keep only the most recent tokens.
        if seq_len > self.max_sequence_length:
            start = seq_len - self.max_sequence_length
            sequence_tensor = sequence_tensor[:, start:, :]
            token_types = token_types[start:]
            seq_len = sequence_tensor.shape[1]
        
        # Add positional embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_embeds = self.position_embedding(positions)
        
        # Add token type embeddings
        token_type_tensor = torch.tensor(token_types, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        type_embeds = self.token_type_embedding(token_type_tensor)
        
        # Combine embeddings
        x = sequence_tensor + pos_embeds + type_embeds
        x = self.dropout(x)
        
        # Create causal attention mask
        causal_mask = self.create_causal_mask(seq_len).to(device)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x, src_mask=causal_mask)
        
        x = self.layer_norm(x)
        
        # Get predictions from the future query slots (the final N positions)
        pred_tokens = x[:, -num_patches_plus_one:, :]  # [batch_size, num_patches+1, embed_dim]
        predicted_features = self.output_head(pred_tokens)
        
        return predicted_features
    
    def predict_uncertainty(self, encoder_features_history, actions, num_samples=10):
        """
        Predict uncertainty using dropout at inference time (Monte Carlo dropout)
        
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