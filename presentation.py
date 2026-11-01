"""Human-readable model names; experiment identifiers remain unchanged."""

def display_names(text):
    return text.replace("YourModel", "Temporal-Patch EEG Transformer").replace("VanillaEEGTransformer", "Vanilla EEG Transformer")
