import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class PolyphaseCodeAnalyzer:
    def __init__(self, code_set: np.ndarray, N: int, L: int):
        # ... existing initialization code ...

    def calculate_ambiguity_function(self, code: np.ndarray, doppler_freqs: np.ndarray, delays: np.ndarray) -> np.ndarray:
        # ... existing ambiguity function code ...

    def plot_doppler_ambiguity(self, velocity_range: float = 100.0, range_max: float = 1000.0, fc: float = 1.0e9, c: float = 3.0e8):
        # ... existing plotting code ...

    def analyze_doppler_ambiguity(self, velocity_range: float = 100.0, fc: float = 1.0e9, c: float = 3.0e8) -> dict:
        # ... existing analysis code ...

    def analyze_noise_effects(self, snr_db_range: np.ndarray = np.linspace(-10, 20, 31)) -> dict:
        # ... existing noise analysis code ...

    def analyze_bandwidth(self) -> dict:
        # ... existing bandwidth analysis code ...

    def plot_analysis_results(self, noise_results: dict, doppler_results: dict, bandwidth_results: dict):
        # ... existing plotting code ... 