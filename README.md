# The Quantum Regression: Rediscovering Planck's Constant

> *"God does not play dice with the universe." — Albert Einstein* > *"Perhaps not, but the sensors do." — The Mystic Engineer*

## 🌌 The Premise

In 1905, Albert Einstein published the paper on the **Photoelectric Effect**, proposing that light consists of discrete packets of energy (quanta). This discovery hinged on a single, fundamental number: **Planck's Constant ($h$)**.

This project is a computational experiment to see if **Artificial Intelligence** can rediscover this fundamental law of the universe.

We do not give the AI the formula. Instead, we feed it noisy, imperfect simulated sensor data—mimicking the chaos of a real-world laboratory. Using **Linear Regression** and **K-Fold Cross-Validation**, the model must extract the signal from the noise and derive the value of $h$ and the Work Function ($\Phi$) of Gold on its own.

---

## 🔬 The Physics (The Hidden Truth)

The model attempts to learn the linear relationship governed by Einstein's Photoelectric Equation:

$$K_{max} = h\nu - \Phi$$

Where:
* **$K_{max}$**: Kinetic Energy of the ejected electron (Target $y$)
* **$h$**: Planck's Constant (Slope $m$)
* **$\nu$**: Frequency of the incident light (Feature $X$)
* **$\Phi$**: Work Function of the metal (Intercept $c$)

**The Challenge:**
Real-world data is messy. Below a certain threshold frequency ($f_0$), no electrons are emitted ($K=0$). This creates a non-linear "hockey stick" shape that confuses standard regression models. The AI must learn to filter the silence to hear the music.

---

## 🏛️ The Architecture

The pipeline is divided into three distinct modules, separating the **Simulation**, the **Analysis**, and the **Visualization**.

### 1. `data_creation.py` (The Genesis)
* **Role:** The Simulator.
* **Function:** Generates synthetic data for Gold ($\Phi = 5.30$ eV). It simulates photons hitting the surface, calculates ideal energies, and then corrupts the data with **Gaussian Noise** ($\sigma = 0.5$ eV) to mimic thermal interference.
* **Output:** `gold_photoelectric_data.csv`

### 2. `model_training.py` (The Analysis)
* **Role:** The Inferential Engine.
* **Function:** Loads the raw data and performs a **5-Fold Cross-Validation**.
* **Logic:** It implements a physics-informed filter (ignoring sub-threshold frequencies) to train a Linear Regression model. It extracts the slope and intercept from each fold to estimate $h$ and $\Phi$.
* **Output:** Console logs detailing the "Discovered Equation" for each fold.

### 3. `model_visualization.py` (The Gallery)
* **Role:** The Presentation Layer.
* **Function:** Generates professional-grade scientific plots using a custom **"Mystic Lavender"** color palette.
* **Output:**
    * **The Triptych:** A 3-panel view showing the Physics, the Model Stability, and the Residual Errors.
    * **The Confidence:** A regression plot with a 95% Confidence Interval shadow.

---

## 📊 The Results

After training on $N=250$ noisy observations, the model achieved the following consensus:

| Constant | True Value (Literature) | AI Predicted Value | Error (%) |
| :--- | :--- | :--- | :--- |
| **Planck's Constant ($h$)** | $6.626 \times 10^{-34}$ J·s | **$6.5374 \times 10^{-34}$ J·s** | **~1.34%** |
| **Work Function ($\Phi$)** | $5.300$ eV | **$5.200$ eV** | **~1.886%** |

*The model successfully identified the physical law despite significant sensor noise.*

---

## 🖼️ The Gallery

### Figure 1: The Analysis Triptych
*A comprehensive view of the signal extraction process.*
*(Place `visual_analysis_triptych.png` here)*

### Figure 2: The Confidence Interval
*Visualizing the certainty of the discovered law.*
*(Place `visual_analysis_confidence.png` here)*

---

## 🛠️ Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Quantum-Quill-314/Lux-Ex-Machina.git]
    cd Lux-Ex-Machina
    ```

2.  **Install dependencies:**
    ```bash
    pip install numpy pandas matplotlib seaborn scikit-learn
    ```

3.  **Run the pipeline:**
    ```bash
    # Step 1: Generate the Universe
    python data_creation.py

    # Step 2: Analyze the Physics
    python model_training.py

    # Step 3: Paint the Results
    python model_visualization.py
    ```

---

## ✒️ Author

**The Mystic Engineer** *Student of AI, Lover of Physics.* > *"Data is just the shadow of the truth."*
