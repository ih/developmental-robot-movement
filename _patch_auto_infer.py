from pathlib import Path

path = Path("session_explorer.py")
text = path.read_text(encoding="utf-8")
needle = "        plt.tight_layout()\n        plt.show()\n\n\n"
replacement = "        plt.tight_layout()\n        plt.show()\n        display_autoencoder_weight_summary(autoencoder, \"Autoencoder weights (current)\")\n\n\n"
if needle not in text:
    raise SystemExit("target block not found for inference")
text = text.replace(needle, replacement, 1)
path.write_text(text, encoding="utf-8")
