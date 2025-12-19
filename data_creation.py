"""
Project: The Quantum Regression (Genesis Module)
Description: 
    The Architect of Chaos. This script simulates the quantum behavior of photons
    striking a Gold surface. It generates synthetic data obeying Einstein's 
    Photoelectric Equation, then corrupts it with Gaussian noise to mimic 
    the imperfect nature of real-world sensors.
"""

import numpy as np
import pandas as pd

# ==========================================
# PHASE 1: The Constants (The Laws of Physics)
# ==========================================
print("\n" + "="*50)
print("  >>> INITIATING REALITY SIMULATION <<<")
print("="*50 + "\n")

print("... Loading Universal Constants ...")

# Fundamental constants
h = 6.626e-34       # Planck's constant (J*s)
work_function_ev = 5.30  # Gold's resistance to letting go (eV)
e_charge = 1.602e-19     # Elementary charge (C)

# Derived threshold
phi = work_function_ev * e_charge  # Work function in Joules
f_threshold = phi / h              # Minimum frequency to liberate an electron

print(f"  > Target Material: Gold (Au)")
print(f"  > Work Function:   {work_function_ev} eV")
print(f"  > Threshold Freq:  {f_threshold:.2e} Hz")
print("-" * 30)

# ==========================================
# PHASE 2: The Simulation (The Light)
# ==========================================
print("... firing photons at the surface ...")

# We ensure reproducibility of our chaos
np.random.seed(19)

# 1. Valid Frequencies (UV/X-ray): Strong enough to eject electrons
valid_frequencies = np.random.uniform(1.28e15, 3.256e15, 200)

# 2. Invalid Frequencies (Visible/IR): Too weak, they just heat the metal
invalid_frequencies = np.random.uniform(0.9857e15, 1.27e15, 50)

# Merge and shuffle the timeline
frequencies = np.concatenate((valid_frequencies, invalid_frequencies))
np.random.shuffle(frequencies)

# Calculate Pure Kinetic Energy (Einstein's Ideal World)
# K_max = h*f - Phi
ke_raw = (h * frequencies) - phi

# The "ReLU" of Physics: Energy cannot be negative. 
# If photon energy < work function, electron stays home (0 Joules).
ke_clean = np.maximum(ke_raw, 0)

print(f"  > Photons Fired: {len(frequencies)}")
print(f"  > Valid Interactions: {len(valid_frequencies)}")
print(f"  > Duds (Below Threshold): {len(invalid_frequencies)}")
print("-" * 30)

# ==========================================
# PHASE 3: The Entropy (The Noise)
# ==========================================
print("... Adding sensor imperfections (Gaussian Noise) ...")

# A real sensor is never perfect. It has thermal noise.
noise_amp_ev = 0.5  # Standard deviation in eV (Adjusted for realism)
noise_amp_joules = noise_amp_ev * e_charge

# Generating the static
noise = np.random.normal(loc=0, scale=noise_amp_joules, size=len(frequencies))

# The Final "observed" data
noisy_ke = ke_clean + noise

# ==========================================
# PHASE 4: The Artifact (Export)
# ==========================================
print("... Crystallizing data into storage ...")

data = {
    'Frequency_Hz': frequencies,
    'Kinetic_Energy_J': noisy_ke
}

df = pd.DataFrame(data)

# Exporting with scientific notation to preserve precision
filename = 'gold_photoelectric_data.csv'
df.to_csv(filename, index=False, float_format='%.2e')

print("\n" + "="*50)
print(f"  >>> DATA GENERATION COMPLETE: {filename} <<<")
print("="*50 + "\n")

# Preview for the curious
print("Artifact Preview:")
print(df.head())

print("\n")
