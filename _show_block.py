from pathlib import Path
text = Path('session_explorer.py').read_text(encoding='utf-8')
idx = text.index('        fig, axes = plt.subplots(1, 2, figsize=(8, 4))')
end = text.index('session_state["autoencoder"] = adaptive_world_model.autoencoder', idx) + len('session_state["autoencoder"] = adaptive_world_model.autoencoder')
print(text[idx:end])
