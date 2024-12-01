import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import random
from fractions import Fraction
import seaborn as sns  # Make sure to import seaborn for heatmap plotting
from multiprocessing import Pool  # Import Pool for parallel processing
import hashlib  # Import hashlib for creating hash keys
import time  # Import the time module
import os


class PolyphaseCodeGA:
    def __init__(self, N: int, L: int, M: int, population_size: int = 100):
        """
        Initialize the GA for generating orthogonal polyphase codes.

        Args:
            N: Length of each code sequence
            L: Number of codes to generate
            M: Number of distinct phases (M=2 for binary, M=4 for quadriphase, etc.)
            population_size: Size of the GA population
        """
        self.N = N
        self.L = L
        self.M = M
        self.population_size = population_size
        self.phases = np.linspace(
            0, 2 * np.pi * (M - 1) / M, M
        )  # Generate M equally spaced phases

        # Create a mapping of numerical values to pi fractions
        self.phase_to_pi = {}
        for phase in self.phases:
            # Convert to fraction of pi
            frac = Fraction(phase / np.pi).limit_denominator()
            self.phase_to_pi[phase] = frac

        # Weights for the cost function
        self.weights = [1.0, 1.0, 1.0, 1.0]  # Initialize weights for Q1, Q2, Q3, Q4

        self.correlation_cache = {}  # Initialize a cache for correlations
        self.correlation_cache_hits = 0  # Initialize cache hit counter
        self.correlation_cache_misses = 0  # Initialize cache miss counter

    def phase_to_string(self, phase: float) -> str:
        """Convert a phase value to a string representation in terms of π."""
        frac = self.phase_to_pi[phase]
        if frac == 0:
            return "0"
        elif frac == 1:
            return "π"
        elif frac.numerator == 1:
            return f"π/{frac.denominator}"
        elif frac.numerator == -1:
            return f"-π/{frac.denominator}"
        else:
            return f"{frac.numerator}π/{frac.denominator}"

    def initialize_population(self) -> List[np.ndarray]:
        """Generate initial random population of code sets."""
        return [
            np.random.choice(self.phases, size=(self.L, self.N))
            for _ in range(self.population_size)
        ]

    def calculate_correlation(
        self, sequence1: np.ndarray, sequence2: np.ndarray
    ) -> np.ndarray:
        """Calculate correlation between two sequences with memoization."""
        key = hashlib.md5((sequence1.tobytes() + sequence2.tobytes())).hexdigest()
        if key in self.correlation_cache:
            self.correlation_cache_hits += 1  # Increment hit counter
            return self.correlation_cache[key]

        self.correlation_cache_misses += 1  # Increment miss counter
        N = len(sequence1)
        correlation = np.zeros(2 * N - 1, dtype=complex)

        # Use the existing thread pool for parallelization
        for k in range(-N + 1, N):
            if k < 0:
                correlation[k + N - 1] = (
                    np.sum(
                        np.exp(1j * sequence1[-k:]) * np.exp(-1j * sequence2[: k + N])
                    )
                    / N
                )
            else:
                correlation[k + N - 1] = (
                    np.sum(
                        np.exp(1j * sequence1[: N - k]) * np.exp(-1j * sequence2[k:])
                    )
                    / N
                )
        correlation = np.abs(correlation)
        self.correlation_cache[key] = correlation
        return correlation

    def _calculate_correlation_for_k(
        self, k: int, sequence1: np.ndarray, sequence2: np.ndarray
    ) -> complex:
        """Calculate correlation for a specific value of k."""
        N = len(sequence1)
        if k < 0:
            return np.sum(
                np.exp(1j * sequence1[-k:]) * np.exp(-1j * sequence2[: k + N]) / N
            )
        else:
            return np.sum(
                np.exp(1j * sequence1[: N - k]) * np.exp(-1j * sequence2[k:]) / N
            )

    def calculate_autocorr_sidelobe_peaks(self, code_set: np.ndarray) -> float:
        """Calculate the sum of autocorrelation sidelobe peaks for the code set."""
        total_peaks = 0
        for i in range(self.L):  # Revert to a simple for-loop
            auto_corr = self.calculate_correlation(code_set[i], code_set[i])
            # Exclude the main peak
            sidelobes = np.concatenate([auto_corr[: self.N - 1], auto_corr[self.N :]])
            total_peaks += np.max(sidelobes)  # Sidelobe peak
        return total_peaks

    def calculate_cross_corr_peaks(self, code_set: np.ndarray) -> float:
        """Calculate the sum of cross-correlation peaks for the code set."""
        total_peaks = 0
        for i in range(self.L):
            for j in range(i + 1, self.L):
                cross_corr = self.calculate_correlation(code_set[i], code_set[j])
                total_peaks += np.max(cross_corr)  # Main peak
        return total_peaks

    def calculate_autocorr_energy(self, code_set: np.ndarray) -> float:
        """Calculate the total autocorrelation sidelobe energy for the code set."""
        total_energy = 0
        for i in range(self.L):
            auto_corr = self.calculate_correlation(code_set[i], code_set[i])
            sidelobes = np.concatenate([auto_corr[: self.N - 1], auto_corr[self.N :]])
            total_energy += np.sum(sidelobes)  # Sidelobe energy
        return total_energy

    def calculate_cross_corr_energy(self, code_set: np.ndarray) -> float:
        """Calculate the total cross-correlation energy for the code set."""
        total_energy = 0
        for i in range(self.L):
            for j in range(i + 1, self.L):
                cross_corr = self.calculate_correlation(code_set[i], code_set[j])
                total_energy += np.sum(cross_corr)  # Cross-correlation energy
        return total_energy

    def fitness(self, code_set: np.ndarray) -> float:
        """
        Calculate fitness of a code set based on autocorrelation and cross-correlation properties.
        Lower score is better.
        """
        Q1 = self.calculate_autocorr_sidelobe_peaks(code_set)
        Q2 = self.calculate_cross_corr_peaks(code_set)
        Q3 = self.calculate_autocorr_energy(code_set)
        Q4 = self.calculate_cross_corr_energy(code_set)

        # Calculate weighted sum using current weights
        total_cost = (
            self.weights[0] * Q1
            + self.weights[1] * Q2
            + self.weights[2] * Q3
            + self.weights[3] * Q4
        )

        return total_cost

    def selection(
        self, population: List[np.ndarray], fitness_scores: List[float]
    ) -> List[np.ndarray]:
        """Select parents using tournament selection."""
        tournament_size = 3
        selected = []

        for _ in range(len(population)):
            tournament_idx = random.sample(range(len(population)), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_idx]
            winner_idx = tournament_idx[np.argmin(tournament_fitness)]
            selected.append(population[winner_idx].copy())

        return selected

    def crossover(
        self, parent1: np.ndarray, parent2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform crossover between two parents."""
        child1 = parent1.copy()
        child2 = parent2.copy()

        if random.random() < 0.8:  # Crossover probability
            for i in range(self.L):
                crossover_point = random.randint(1, self.N - 1)
                child1[i, crossover_point:] = parent2[i, crossover_point:]
                child2[i, crossover_point:] = parent1[i, crossover_point:]

        return child1, child2

    def mutation(self, code_set: np.ndarray) -> np.ndarray:
        """Apply mutation to a code set."""
        mutated = code_set.copy()
        mutation_rate = 0.1

        for i in range(self.L):
            for j in range(self.N):
                if random.random() < mutation_rate:
                    mutated[i, j] = random.choice(self.phases)

        return mutated

    def optimize(self, generations: int = 100) -> Tuple[np.ndarray, float]:
        """Run the genetic algorithm optimization with adaptive weights."""
        population = self.initialize_population()
        best_fitness = float("inf")
        best_solution = np.zeros((self.L, self.N))  # Initialize with a valid ndarray
        best_fitness_history = []  # Record best fitness for each generation
        min_fitness_history = []  # Record minimum fitness for each generation
        total_time = 0  # Initialize total time for generations

        with Pool() as pool:  # Use Pool instead of ThreadPoolExecutor
            for gen in range(generations):
                start_time = time.time()  # Start time for the generation

                # Evaluate fitness using Pool for parallelization
                fitness_scores = pool.map(self.fitness, population)

                # Track best solution
                min_fitness = min(fitness_scores)  # Minimum fitness of the population
                min_fitness_history.append(min_fitness)  # Record minimum fitness

                min_fitness_idx = np.argmin(fitness_scores)
                if fitness_scores[min_fitness_idx] < best_fitness:
                    best_fitness = fitness_scores[min_fitness_idx]
                    best_solution = population[min_fitness_idx].copy()
                    print(f"Generation {gen}: New best fitness = {best_fitness}")
                    best_fitness_history.append(best_fitness)  # Record best fitness
                else:
                    best_fitness_history.append(
                        best_fitness_history[-1]
                    )  # Keep previous best

                # Update weights based on fitness scores
                self.update_weights(fitness_scores)

                # Selection
                selected = self.selection(population, fitness_scores)

                # Create new population
                new_population = []
                for i in range(0, self.population_size, 2):
                    parent1 = selected[i]
                    parent2 = selected[min(i + 1, self.population_size - 1)]

                    # Crossover
                    child1, child2 = self.crossover(parent1, parent2)

                    # Mutation
                    child1 = self.mutation(child1)
                    child2 = self.mutation(child2)

                    new_population.extend([child1, child2])

                population = new_population[: self.population_size]

                end_time = time.time()  # End time for the generation
                generation_time = (
                    end_time - start_time
                )  # Time taken for this generation
                total_time += generation_time  # Accumulate total time

        # Calculate average time per generation
        average_time_per_generation = total_time / generations
        print(f"Average time per generation: {average_time_per_generation:.4f} seconds")

        # Plot best fitness history after optimization
        plt.figure(figsize=(10, 5))
        plt.plot(
            min_fitness_history, label="Minimum Fitness Score", marker=""
        )  # No dots
        plt.plot(
            best_fitness_history, label="Best Fitness", marker="o"
        )  # Dots for new best
        plt.title("Fitness Over Generations")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.legend()
        plt.grid(True)
        plt.show()
        return best_solution, best_fitness

    def update_weights(self, fitness_scores: List[float]):
        """Update weights based on fitness scores."""
        # Calculate relative magnitudes and adjust weights inversely
        total_fitness = sum(
            1 / f for f in fitness_scores
        )  # Use inverse fitness for minimization
        relative_magnitudes = [(1 / f) / total_fitness for f in fitness_scores]

        # Update weights to give more importance to terms with smaller relative magnitude
        sum_inv_magnitudes = sum(1 / m if m > 0 else 1.0 for m in relative_magnitudes)
        new_weights = [
            (1 / m if m > 0 else 1.0) / sum_inv_magnitudes for m in relative_magnitudes
        ]

        # Smooth weight updates using moving average
        alpha = 0.3  # Weight update rate
        self.weights = [
            alpha * nw + (1 - alpha) * w for w, nw in zip(self.weights, new_weights)
        ]

    def plot_correlations(self, code_set: np.ndarray):
        """Plot autocorrelation and cross-correlation functions for the code set."""

        print(f"Cache hits: {self.correlation_cache_hits}")
        print(f"Cache misses: {self.correlation_cache_misses}")

        n_plots = self.L + (self.L * (self.L - 1)) // 2
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots))

        plot_idx = 0

        # Plot autocorrelations
        for i in range(self.L):
            auto_corr = self.calculate_correlation(code_set[i], code_set[i])
            axes[plot_idx].plot(auto_corr)
            axes[plot_idx].set_title(f"Autocorrelation of Code {i+1}")
            axes[plot_idx].grid(True)
            plot_idx += 1

        # Plot cross-correlations
        for i in range(self.L):
            for j in range(i + 1, self.L):
                cross_corr = self.calculate_correlation(code_set[i], code_set[j])
                axes[plot_idx].plot(cross_corr)
                axes[plot_idx].set_title(f"Cross-correlation of Codes {i+1} and {j+1}")
                axes[plot_idx].grid(True)
                plot_idx += 1

        plt.tight_layout()
        plt.show()

    def print_codes(self, code_set: np.ndarray):
        """Print the codes with phases in terms of π."""
        print("\nBest codes found (phases in terms of π):")
        for i, code in enumerate(code_set):
            print(f"\nCode {i+1}:")
            phase_strings = [self.phase_to_string(phase) for phase in code]
            print("[" + ", ".join(phase_strings) + "]")

    def calculate_autocorr_sidelobe_peaks_matrix(
        self, code_set: np.ndarray
    ) -> np.ndarray:
        """Calculate the autocorrelation sidelobe peaks for all codes and return a matrix."""
        matrix = np.zeros((self.L, self.L))
        for i in range(self.L):
            for j in range(self.L):
                if i == j:
                    auto_corr = self.calculate_correlation(code_set[i], code_set[i])
                    sidelobes = np.concatenate(
                        [auto_corr[: self.N - 1], auto_corr[self.N :]]
                    )
                    matrix[i, j] = np.max(sidelobes)  # Sidelobe peak
                else:
                    matrix[i, j] = 0  # Cross-correlation will be handled separately
        return matrix

    def calculate_cross_corr_peaks_matrix(self, code_set: np.ndarray) -> np.ndarray:
        """Calculate the cross-correlation peaks for all codes and return a matrix."""
        matrix = np.zeros((self.L, self.L))
        for i in range(self.L):
            for j in range(i + 1, self.L):
                cross_corr = self.calculate_correlation(code_set[i], code_set[j])
                matrix[i, j] = np.max(cross_corr)  # Main peak
                matrix[j, i] = matrix[i, j]  # Symmetric
        return matrix

    def plot_combined_heatmap(self, code_set: np.ndarray):
        """Plot a combined heatmap for autocorrelation sidelobe peaks and cross-correlation peaks."""
        # Calculate matrices
        auto_corr_matrix = self.calculate_autocorr_sidelobe_peaks_matrix(code_set)
        cross_corr_matrix = self.calculate_cross_corr_peaks_matrix(code_set)

        # Create a combined matrix
        combined_matrix = np.zeros((self.L, self.L))
        combined_matrix += (
            auto_corr_matrix  # Fill diagonal with autocorrelation sidelobe peaks
        )
        combined_matrix += (
            cross_corr_matrix  # Fill off-diagonal with cross-correlation peaks
        )

        # Normalize combined matrix to range [0, 1]
        combined_matrix = (
            combined_matrix / np.max(combined_matrix)
            if np.max(combined_matrix) > 0
            else combined_matrix
        )

        # Plotting heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            combined_matrix,
            cmap="viridis",
            cbar=True,
            annot=True,
            fmt=".2f",
            xticklabels=[f"Code {i+1}" for i in range(self.L)],
            yticklabels=[f"Code {i+1}" for i in range(self.L)],
        )
        plt.title(
            "Combined Heatmap of Autocorrelation Sidelobe Peaks and Cross-Correlation Peaks"
        )
        plt.tight_layout()
        plt.show()


class PolyphaseCodeAnalyzer:
    def __init__(self, code_set: np.ndarray, N: int, L: int):
        """
        Initialize the analyzer for polyphase codes.

        Args:
            code_set: Array of polyphase codes to analyze
            N: Length of each code sequence
            L: Number of codes
        """
        self.code_set = code_set
        self.N = N
        self.L = L

    def calculate_ambiguity_function(
        self, code: np.ndarray, doppler_freqs: np.ndarray, delays: np.ndarray
    ) -> np.ndarray:
        """
        Calculate the ambiguity function for a given code.

        Args:
            code: Complex envelope of the code sequence
            doppler_freqs: Array of Doppler frequencies to evaluate
            delays: Array of delay values to evaluate

        Returns:
            2D array containing ambiguity function values
        """
        complex_code = np.exp(1j * code)
        ambiguity = np.zeros((len(doppler_freqs), len(delays)), dtype=complex)

        for i, fd in enumerate(doppler_freqs):
            doppler_term = np.exp(2j * np.pi * fd * np.arange(self.N))
            modified_code = complex_code * doppler_term

            for j, delay in enumerate(delays):
                if delay >= 0:
                    # Positive delay
                    if delay < self.N:
                        corr = np.sum(
                            modified_code[: -int(delay)]
                            * np.conj(complex_code[int(delay) :])
                        )
                        ambiguity[i, j] = corr / self.N
                else:
                    # Negative delay
                    if abs(delay) < self.N:
                        corr = np.sum(
                            modified_code[int(-delay) :]
                            * np.conj(complex_code[: int(self.N + delay)])
                        )
                        ambiguity[i, j] = corr / self.N

        return np.abs(ambiguity)

    def plot_doppler_ambiguity(
        self,
        velocity_range: float = 100.0,  # m/s
        range_max: float = 1000.0,  # meters
        fc: float = 1.0e9,  # carrier frequency (Hz)
        c: float = 3.0e8,  # speed of light (m/s)
    ):
        """
        Plot the Doppler ambiguity function for all codes.

        Args:
            velocity_range: Maximum velocity to plot (±velocity_range)
            range_max: Maximum range to plot
            fc: Carrier frequency in Hz
            c: Speed of light in m/s
        """
        # Calculate Doppler frequency range
        fd_max = 2 * velocity_range * fc / c
        doppler_freqs = (
            np.linspace(-fd_max, fd_max, 201) / self.N
        )  # Normalized frequencies

        # Calculate delay range
        delays = np.linspace(-self.N / 2, self.N / 2, 201)

        # Create velocity and range axes for plotting
        velocities = doppler_freqs * c * self.N / (2 * fc)  # Convert to velocity
        ranges = delays * c / (2 * fc)  # Convert to range

        # Plot ambiguity function for each code
        for i in range(self.L):
            # Calculate ambiguity function
            amb_func = self.calculate_ambiguity_function(
                self.code_set[i], doppler_freqs, delays
            )

            # Convert to dB
            amb_func_db = 20 * np.log10(
                amb_func + 1e-10
            )  # Add small value to avoid log(0)
            amb_func_db = np.clip(amb_func_db, -60, 0)  # Clip at -60 dB

            # Create 3D plot
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection="3d")

            # Create meshgrid for plotting
            V, R = np.meshgrid(velocities, ranges)

            # Plot surface
            surf = ax.plot_surface(
                V, R, amb_func_db.T, cmap="viridis", linewidth=0, antialiased=True
            )

            # Add colorbar
            fig.colorbar(surf, ax=ax, label="Amplitude (dB)")

            # Set labels and title
            ax.set_xlabel("Velocity (m/s)")
            ax.set_ylabel("Range (m)")
            ax.set_zlabel("Amplitude (dB)")
            ax.set_title(f"Ambiguity Function - Code {i+1}")

            # Adjust view angle
            ax.view_init(elev=30, azim=45)

            plt.tight_layout()
            plt.show()

            # Also plot 2D heatmap
            plt.figure(figsize=(12, 8))
            plt.pcolormesh(
                velocities, ranges, amb_func_db.T, cmap="viridis", shading="auto"
            )
            plt.colorbar(label="Amplitude (dB)")
            plt.xlabel("Velocity (m/s)")
            plt.ylabel("Range (m)")
            plt.title(f"Ambiguity Function (2D) - Code {i+1}")
            plt.tight_layout()
            plt.show()

    def analyze_doppler_ambiguity(
        self, velocity_range: float = 100.0, fc: float = 1.0e9, c: float = 3.0e8
    ) -> dict:
        """
        Analyze Doppler ambiguity properties of the codes.

        Returns:
            Dictionary containing Doppler tolerance metrics
        """
        # Calculate Doppler frequency range
        fd_max = 2 * velocity_range * fc / c
        doppler_freqs = np.linspace(-fd_max, fd_max, 201) / self.N
        delays = np.array([0])  # Only analyze zero delay

        results = {
            "doppler_shifts": doppler_freqs * self.N,  # Convert back to Hz
            "ambiguity_function": [],
            "doppler_tolerance": [],
        }

        for i in range(self.L):
            amb_func = self.calculate_ambiguity_function(
                self.code_set[i], doppler_freqs, delays
            )

            # Store zero-delay cut of ambiguity function
            results["ambiguity_function"].append(amb_func[:, 0])

            # Calculate Doppler tolerance (width at -3dB)
            amb_func_db = 20 * np.log10(amb_func[:, 0])
            tolerance_mask = amb_func_db >= -3
            results["doppler_tolerance"].append(
                np.sum(tolerance_mask) * (2 * fd_max / 200)
            )

        return results

    def analyze_noise_effects(
        self, snr_db_range: np.ndarray = np.linspace(-10, 20, 31)
    ) -> dict:
        """
        Analyze the effect of noise on code detection performance.

        Args:
            snr_db_range: Array of SNR values in dB to analyze

        Returns:
            Dictionary containing detection probabilities and false alarm rates
        """
        results = {
            "snr_db": snr_db_range,
            "detection_prob": [],
            "false_alarm_rate": [],
            "correlation_degradation": [],
        }

        for snr_db in snr_db_range:
            # Convert SNR from dB to linear scale
            snr_linear = 10 ** (snr_db / 10)

            # Generate noise samples
            noise_power = 1 / snr_linear
            noise = np.sqrt(noise_power / 2) * (
                np.random.randn(self.L, self.N) + 1j * np.random.randn(self.L, self.N)
            )

            # Add noise to the complex envelope of the codes
            noisy_codes = np.exp(1j * self.code_set) + noise

            # Calculate correlation properties with noise
            detection_prob = 0
            false_alarms = 0
            correlation_degradation = 0

            for i in range(self.L):
                # Calculate autocorrelation with noise
                auto_corr_noisy = np.correlate(
                    noisy_codes[i], np.exp(1j * self.code_set[i]), mode="full"
                )
                auto_corr_clean = np.correlate(
                    np.exp(1j * self.code_set[i]),
                    np.exp(1j * self.code_set[i]),
                    mode="full",
                )

                # Detection probability (normalized peak height > threshold)
                threshold = 0.5  # Adjustable threshold
                peak_idx = len(auto_corr_noisy) // 2
                detection_prob += np.abs(auto_corr_noisy[peak_idx]) / self.N > threshold

                # False alarm rate (sidelobes exceeding threshold)
                sidelobes = np.concatenate(
                    [auto_corr_noisy[:peak_idx], auto_corr_noisy[peak_idx + 1 :]]
                )
                false_alarms += np.sum(np.abs(sidelobes) / self.N > threshold)

                # Correlation degradation
                correlation_degradation += (
                    np.mean(np.abs(auto_corr_noisy - auto_corr_clean)) / self.N
                )

            results["detection_prob"].append(detection_prob / self.L)
            results["false_alarm_rate"].append(
                false_alarms / (self.L * (2 * self.N - 2))
            )
            results["correlation_degradation"].append(correlation_degradation / self.L)

        return results

    def analyze_bandwidth(self) -> dict:
        """
        Analyze bandwidth properties of the codes.

        Returns:
            Dictionary containing bandwidth metrics
        """
        results = {
            "rms_bandwidth": [],
            "fractional_bandwidth": [],
            "spectral_peaks": [],
        }

        for i in range(self.L):
            # Calculate power spectral density
            code = np.exp(1j * self.code_set[i])
            freq = np.fft.fftfreq(self.N)
            psd = np.abs(np.fft.fft(code)) ** 2

            # RMS bandwidth
            rms_bw = np.sqrt(np.sum(freq**2 * psd) / np.sum(psd))
            results["rms_bandwidth"].append(rms_bw)

            # Fractional bandwidth (bandwidth containing 90% of energy)
            sorted_psd = np.sort(psd)[::-1]
            cumsum_psd = np.cumsum(sorted_psd) / np.sum(psd)
            bw_90 = np.sum(cumsum_psd <= 0.9) / self.N
            results["fractional_bandwidth"].append(bw_90)

            # Peak-to-average power ratio in frequency domain
            results["spectral_peaks"].append(np.max(psd) / np.mean(psd))

        return results

    def plot_analysis_results(
        self, noise_results: dict, doppler_results: dict, bandwidth_results: dict
    ):
        """
        Plot comprehensive analysis results.
        """
        plt.figure(figsize=(15, 10))

        # Plot 1: Noise Effects
        plt.subplot(2, 2, 1)
        plt.plot(
            noise_results["snr_db"],
            noise_results["detection_prob"],
            "b-",
            label="Detection Probability",
        )
        plt.plot(
            noise_results["snr_db"],
            noise_results["false_alarm_rate"],
            "r--",
            label="False Alarm Rate",
        )
        plt.grid(True)
        plt.xlabel("SNR (dB)")
        plt.ylabel("Probability")
        plt.legend()
        plt.title("Noise Effects on Detection Performance")

        # Plot 2: Doppler Ambiguity
        plt.subplot(2, 2, 2)
        for i in range(self.L):
            plt.plot(
                doppler_results["doppler_shifts"],
                doppler_results["ambiguity_function"][i],
                label=f"Code {i+1}",
            )
        plt.grid(True)
        plt.xlabel("Normalized Doppler Frequency")
        plt.ylabel("Ambiguity Function")
        plt.legend()
        plt.title("Doppler Ambiguity Function")

        # Plot 3: Bandwidth Properties
        plt.subplot(2, 2, 3)
        x = np.arange(self.L) + 1
        width = 0.25
        plt.bar(x - width, bandwidth_results["rms_bandwidth"], width, label="RMS BW")
        plt.bar(x, bandwidth_results["fractional_bandwidth"], width, label="90% BW")
        plt.bar(
            x + width,
            np.array(bandwidth_results["spectral_peaks"])
            / np.max(bandwidth_results["spectral_peaks"]),
            width,
            label="Normalized PAPR",
        )
        plt.xlabel("Code Number")
        plt.ylabel("Normalized Value")
        plt.legend()
        plt.title("Bandwidth Properties")

        # Plot 4: Correlation Degradation vs SNR
        plt.subplot(2, 2, 4)
        plt.plot(noise_results["snr_db"], noise_results["correlation_degradation"])
        plt.grid(True)
        plt.xlabel("SNR (dB)")
        plt.ylabel("Correlation Degradation")
        plt.title("Correlation Degradation vs SNR")

        plt.tight_layout()
        plt.show()


# Example usage
def main():
    # Parameters
    N = 40  # Code length
    L = 4  # Number of codes
    M = 4  # Number of phases (4 for quadriphase)

    # Create and run optimizer
    optimizer = PolyphaseCodeGA(N=N, L=L, M=M, population_size=100)
    best_codes, best_fitness = optimizer.optimize(generations=100)

    # Print results
    optimizer.print_codes(best_codes)
    print(f"\nFinal fitness score: {best_fitness}")

    # Plot correlations
    optimizer.plot_correlations(best_codes)

    # Plot combined heatmap
    optimizer.plot_combined_heatmap(best_codes)

    # Analyze code properties
    analyzer = PolyphaseCodeAnalyzer(best_codes, N, L)

    # Run analyses
    noise_results = analyzer.analyze_noise_effects()
    doppler_results = analyzer.analyze_doppler_ambiguity()
    bandwidth_results = analyzer.analyze_bandwidth()

    # Plot analysis results
    analyzer.plot_analysis_results(noise_results, doppler_results, bandwidth_results)

    # Print summary statistics
    print("\nAnalysis Summary:")
    print(
        "Doppler Tolerance (normalized):", np.mean(doppler_results["doppler_tolerance"])
    )
    print("Average RMS Bandwidth:", np.mean(bandwidth_results["rms_bandwidth"]))
    print("Average 90% Bandwidth:", np.mean(bandwidth_results["fractional_bandwidth"]))
    print("Average Spectral PAPR:", np.mean(bandwidth_results["spectral_peaks"]))

    # Plot Doppler ambiguity
    analyzer.plot_doppler_ambiguity(
        velocity_range=100.0,  # ±100 m/s
        range_max=1000.0,  # 1000 meters
        fc=1.0e9,  # 1 GHz carrier
        c=3.0e8,  # Speed of light
    )

    # Get Doppler analysis results
    doppler_results = analyzer.analyze_doppler_ambiguity()


if __name__ == "__main__":
    main()
