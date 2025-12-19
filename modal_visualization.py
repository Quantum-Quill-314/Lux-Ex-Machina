"""
Project: The Quantum Regression (Visualization Module)
Author: The Mystic Engineer
Theme: The Lavender Ether
Description: 
    The Gallery. This script creates the "Triptych" and "Confidence" artifacts.
    It uses a custom Purple/Lavender palette to visualize the relationship 
    between Frequency and Kinetic Energy.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

# ==========================================
# CONFIGURATION (The Aesthetics)
# ==========================================
# Setting the dark grid for that "Cyber-Mystic" look
plt.style.use('seaborn-v0_8-darkgrid')

# The Mystic Color Palette 💜
PALETTE = {
    'active': '#4B0082',    # Indigo/Deep Purple (The Signal)
    'line':   '#FF00FF',    # Magenta/Neon Pink (The Physics)
    'dud':    '#E6E6FA',    # Lavender (The Noise/Ghost data)
    'thresh': '#00FFFF',    # Cyan (The Threshold Marker)
    'shadow': '#9370DB'     # Medium Purple (Confidence Interval)
}

# ==========================================
# PHASE 1: Load & Prep
# ==========================================
print("\n" + "="*50)
print("  >>> INITIATING VISUALIZATION PROTOCOL (LAVENDER MODE) <<<")
print("="*50)

df = pd.read_csv('gold_photoelectric_data.csv')
X = df['Frequency_Hz'].values.reshape(-1, 1)
y = df['Kinetic_Energy_J'].values

# The Physics Threshold (Gold)
THRESHOLD_FREQ = 1.28e15 

print("... Data Loaded. Palette mixed ...")

# ==========================================
# PHASE 2: FIGURE 1 - THE TRIPTYCH
# ==========================================
print("... Painting The Triptych ...")

fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# --- SUBPLOT A: The Grand Unified Plot (Physics) ---

# 1. The "Ghost" Data (Below Threshold)
mask_dud = X.flatten() <= THRESHOLD_FREQ
ax1.scatter(X[mask_dud], y[mask_dud], color=PALETTE['dud'], alpha=0.6, label='Sub-Threshold (Noise)')

# 2. The "Real" Data (Above Threshold)
mask_active = X.flatten() > THRESHOLD_FREQ
ax1.scatter(X[mask_active], y[mask_active], color=PALETTE['active'], edgecolor='white', linewidth=0.5, label='Ejected Electrons')

# 3. The Master Model (Retrained here for visualization)
model_master = LinearRegression()
model_master.fit(X[mask_active], y[mask_active])
y_pred_active = model_master.predict(X[mask_active])

# Sort for a clean line
sort_idx = np.argsort(X[mask_active].flatten())
ax1.plot(X[mask_active][sort_idx], y_pred_active[sort_idx], color=PALETTE['line'], linewidth=2.5, linestyle='--', label='Regression Model')

# 4. The Threshold Marker
ax1.axvline(THRESHOLD_FREQ, color=PALETTE['thresh'], linestyle=':', linewidth=2, label='Threshold ($f_0$)')

ax1.set_title("A. The Physics (Photoelectric Effect)", fontsize=12, fontweight='bold', color=PALETTE['active'])
ax1.set_xlabel("Frequency (Hz)")
ax1.set_ylabel("Kinetic Energy (J)")
ax1.legend(facecolor='white', framealpha=0.9)
ax1.ticklabel_format(style='sci', axis='both', scilimits=(0,0))


# --- SUBPLOT B: The Spiderweb (Stability) ---
# We recreate the K-Fold split visually
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Background ghosts
ax2.scatter(X, y, color=PALETTE['dud'], alpha=0.4) 

# Generate 5 shades of purple for the 5 folds
purple_shades = sns.color_palette("light:purple", n_colors=8)[3:] 

fold_i = 0
for train_idx, _ in kf.split(X):
    # Re-isolate the active physics for this fold
    X_fold = X[train_idx]
    y_fold = y[train_idx]
    mask_fold = X_fold.flatten() > THRESHOLD_FREQ
    
    # Train fold model
    model_fold = LinearRegression()
    model_fold.fit(X_fold[mask_fold], y_fold[mask_fold])
    
    # Draw line across the screen
    x_range = np.linspace(THRESHOLD_FREQ, X.max(), 100).reshape(-1, 1)
    y_range_pred = model_fold.predict(x_range)
    
    ax2.plot(x_range, y_range_pred, color=purple_shades[fold_i], alpha=0.8, linewidth=1.5, label=f'Fold {fold_i+1}')
    fold_i += 1

ax2.set_title("B. The Stability (5-Fold Cross Validation)", fontsize=12, fontweight='bold', color=PALETTE['active'])
ax2.set_xlabel("Frequency (Hz)")
ax2.set_yticks([]) # Clean look
ax2.legend(fontsize=8, loc='upper left')


# --- SUBPLOT C: The Ghost Hunter (Residuals) ---
residuals = y[mask_active] - y_pred_active

ax3.scatter(X[mask_active], residuals, color=PALETTE['active'], alpha=0.6, edgecolor='white', linewidth=0.5)
ax3.axhline(0, color=PALETTE['line'], linestyle='--', linewidth=1)
ax3.set_title("C. The Residuals (Gaussian Noise)", fontsize=12, fontweight='bold', color=PALETTE['active'])
ax3.set_xlabel("Frequency (Hz)")
ax3.set_ylabel("Residual Error (J)")
ax3.ticklabel_format(style='sci', axis='both', scilimits=(0,0))

plt.tight_layout()
plt.savefig('visual_analysis_triptych.png', dpi=300)
print("  > Artifact Saved: 'visual_analysis_triptych.png'")


# ==========================================
# PHASE 3: FIGURE 2 - THE CONFIDENCE
# ==========================================
print("... Weaving the Confidence Interval ...")

plt.figure(figsize=(10, 6))

df_active = df[df['Frequency_Hz'] > THRESHOLD_FREQ]

# Seaborn Regplot (The easy way to do confidence intervals)
# We map our custom palette to seaborn arguments
sns.regplot(
    x='Frequency_Hz', 
    y='Kinetic_Energy_J', 
    data=df_active, 
    scatter_kws={'color': PALETTE['active'], 'edgecolor': 'white', 'alpha': 0.7, 's': 50},
    line_kws={'color': PALETTE['line'], 'linestyle': '--', 'linewidth': 2},
    ci=95 # The 95% Confidence Shadow
)

plt.title("Regression Model with 95% Confidence Interval", fontsize=14, fontweight='bold', color=PALETTE['active'])
plt.xlabel("Frequency (Hz)", fontsize=12)
plt.ylabel("Kinetic Energy (J)", fontsize=12)

# Add threshold line for context
plt.axvline(THRESHOLD_FREQ, color=PALETTE['thresh'], linestyle=':', linewidth=2, label='Threshold')

plt.legend()
plt.grid(True, alpha=0.3)
plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))

plt.tight_layout()
plt.savefig('visual_analysis_confidence.png', dpi=300)
print("  > Artifact Saved: 'visual_analysis_confidence.png'")

print("\n" + "="*50)
print("  >>> VISUALIZATION COMPLETE <<<")
print("="*50 + "\n")