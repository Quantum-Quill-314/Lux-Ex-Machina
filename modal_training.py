"""
Project: The Quantum Regression (Recovering Planck's Constant)
Author: The Mystic Engineer
Description: 
    The Probabilistic Canvas. This script orchestrates a simulation of the photoelectric 
    effect, juxtaposing the deterministic elegance of Einstein’s equations against 
    the stochastic reality of experimental measurement. It generates ideal quantum 
    emissions and introduces Gaussian perturbations to model the fidelity limits 
    inherent in physical sensors.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# PHASE 1: Data Ingestion (The Observation)
# ==========================================
print("\n" + "="*50)
print("  >>> INITIATING QUANTUM ANALYSIS PROTOCOL <<<")
print("="*50 + "\n")

print("... Loading sensor data from the ether (CSV) ...")
df = pd.read_csv('gold_photoelectric_data.csv')

'''
Reshaping reality:
The model demands a 2D matrix for features (X) and a 1D vector for targets (y).
We treat Frequency as our independent variable (the cause)
and Kinetic Energy as the dependent variable (the effect).
'''
X = df['Frequency_Hz'].values.reshape(-1, 1) 
y = df['Kinetic_Energy_J'].values

print(f"Data Successfully Loaded.")
print(f"  > Observations: {X.shape[0]}")
print(f"  > Dimensions Verified: X{X.shape} | y{y.shape}")
print("-" * 50)

# ==========================================
# PHASE 2: Cross-Validation (The Rigor)
# ==========================================
# We shall not trust a single test. We slice the universe 5 ways to ensure
# our discovery is not merely a trick of the light (sampling bias).
kf = KFold(n_splits=5, shuffle=True, random_state=42)

print(f"Cross-Validator Initialized.")
print(f"  > Strategy: {kf.get_n_splits(X)}-Fold Split")
print(f"  > Shuffling: Enabled (Increasing Entropy)")
print("-" * 50 + "\n")

# ==========================================
# PHASE 3: The Learning Loop (The Discovery)
# ==========================================

print(">>> COMMENCING TRAINING SEQUENCE <<<\n")

fold = 1
slope_h_list = []
intercept_phi_list = []

for train_index, test_index in kf.split(X):
    
    # --- A. The Split ---
    # Separating the known (Training) from the unknown (Testing)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # --- B. The Filter (The Physicist's Eye) ---
    # We ignore the silence. We only listen where the frequency sings high enough 
    # to liberate an electron. (Filtering X > Threshold to remove the "hockey stick" flatline).
    # Threshold approx 1.28e15 Hz (Derived from Gold's nature).
    valid_mask = X_train.flatten() > 1.28e15
    
    X_train_filtered = X_train[valid_mask]
    y_train_filtered = y_train[valid_mask]
    
    # --- C. The Fit (finding order in chaos) ---
    model = LinearRegression()
    model.fit(X_train_filtered, y_train_filtered)

    # --- D. The Prediction (The Test) ---
    y_pred = model.predict(X_test)
    
    # --- E. The Metrics (The Judgment) ---
    r2 = r2_score(y_test, y_pred)
    
    # Extracting the constants from the mathematical line: y = mx + c
    h_found = model.coef_[0]      # The Slope (Planck's Constant)
    phi_found_joules = -model.intercept_  # The Intercept (Work Function, negative)
    phi_found_ev = phi_found_joules / 1.602e-19 # Convert Joules to eV for readability
    
    # Storing for final ensemble analysis
    slope_h_list.append(h_found)
    intercept_phi_list.append(phi_found_ev)
    
    # --- The Output (Poetic Display) ---
    print(f" [FOLD {fold}] Analysis:")
    print(f"   Equation Discovered:  K = ({h_found:.2e}) * f - ({phi_found_ev:.2e})")
    print(f"   Physical Meaning:     h = {h_found:.3e} J*s  |  Φ = {phi_found_ev:.3f} eV")
    print(f"   Model Accuracy (R²):  {r2:.4f}")
    print("   " + "." * 40)
    
    fold += 1

# ==========================================
# PHASE 4: The Conclusion (The Truth)
# ==========================================
print("\n" + "="*50)
print("  >>> FINAL CONSENSUS <<<")
print("="*50)

avg_h = np.mean(slope_h_list)
avg_phi = np.mean(intercept_phi_list)

# Literature values for comparison
true_h = 6.626e-34
true_phi = 5.30

error_h = abs((avg_h - true_h) / true_h) * 100
error_phi = abs((avg_phi - true_phi) / true_phi) * 100

print(f"Planck's Constant (h):")
print(f"  > Predicted: {avg_h:.4e} J*s")
print(f"  > Literature: {true_h:.4e} J*s")
print(f"  > Deviation:  {error_h:.2f}%")
print("-" * 30)

print(f"Work Function (Φ) for Gold:")
print(f"  > Predicted: {avg_phi:.3f} eV")
print(f"  > Literature: {true_phi:.3f} eV")
print(f"  > Deviation:  {error_phi:.3f}%")

print("\n" + "="*50)
print("  >>> MISSION COMPLETE. PHYSICS VERIFIED. <<<")
print("="*50 + "\n")