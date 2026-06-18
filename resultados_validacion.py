import pickle
import numpy as np
import matplotlib.pyplot as plt
# =====================================================
# LOAD DEBUG DATA
# =====================================================
with open(
    "validation_debug_data.pkl",
    "rb"
) as f:
    results = pickle.load(f)
# =====================================================
# SELECT SAMPLE
# =====================================================
sample_idx = np.random.randint(len(results))
sample = results[sample_idx]
# =====================================================
# EXTRACT DATA
# =====================================================
validation_unit = sample["validation_unit"]
matched_unit = sample["matched_unit"]
validation_hi = np.array(
    sample["validation_hi"]
)
matched_curve = np.array(
    sample["matched_curve"]
)
tau = sample["tau"]
actual_rul = sample["actual_rul"]
predicted_rul = sample["predicted_rul"]
similarity = sample["similarity"]
# =====================================================
# ERROR METRICS
# Error normalizado respecto a la vida TOTAL del motor
# (ventana observada + RUL real), no solo respecto al
# RUL real. Esto evita que el porcentaje se infle cuando
# el motor está muy cerca del fallo (actual_rul pequeño).
# =====================================================
window_length = len(validation_hi)
total_life = window_length + actual_rul

error = predicted_rul - actual_rul
percent_error = (error / total_life) * 100
# =====================================================
# AXES
# =====================================================
window_x = np.arange(
    tau,
    tau + window_length
)
failure_cycle = len(
    matched_curve
) - 1
predicted_failure = (
    tau
    + window_length
    + predicted_rul
)
actual_failure = (
    tau
    + window_length
    + actual_rul
)
# =====================================================
# PLOT
# =====================================================
plt.figure(figsize=(14,8))
# -----------------------------------------
# Entire fitted training curve
# -----------------------------------------
plt.plot(
    matched_curve,
    linewidth=2,
    label=f"Matched fitted curve (Unit {matched_unit})"
)
# -----------------------------------------
# Validation window
# -----------------------------------------
plt.plot(
    window_x,
    validation_hi,
    linewidth=4,
    label=f"Validation window (Unit {validation_unit})"
)
# -----------------------------------------
# Window start
# -----------------------------------------
plt.axvline(
    tau,
    linestyle="--",
    alpha=0.7,
    label=f"τ = {tau}"
)
# -----------------------------------------
# Window end
# -----------------------------------------
plt.axvline(
    tau + window_length - 1,
    linestyle="--",
    alpha=0.7
)
# -----------------------------------------
# Actual failure estimate
# -----------------------------------------
plt.axvline(
    actual_failure,
    linestyle=":",
    linewidth=3,
    label=f"Actual failure ({actual_rul:.1f} cycles)"
)
# -----------------------------------------
# Predicted failure estimate
# -----------------------------------------
plt.axvline(
    predicted_failure,
    linestyle="-.",
    linewidth=3,
    label=f"Predicted failure ({predicted_rul:.1f} cycles)"
)
# -----------------------------------------
# True end of matched library curve
# -----------------------------------------
plt.axvline(
    failure_cycle,
    color="black",
    linewidth=2,
    alpha=0.4,
    label="Library failure cycle"
)
# =====================================================
# LABELS
# =====================================================
plt.title(
    f"Validation Unit {validation_unit} "
    f"vs Matched Unit {matched_unit}\n"
    f"Similarity = {similarity:.4f} | "
    f"Error = {error:+.1f} cycles ({percent_error:+.2f}% of total life)"
)
plt.xlabel("Cycle")
plt.ylabel("Health Indicator")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
print(f"Actual RUL: {actual_rul:.1f} cycles")
print(f"Predicted RUL: {predicted_rul:.1f} cycles")
print(f"Total life observed: {total_life:.1f} cycles")
print(f"Error: {error:+.1f} cycles ({percent_error:+.2f}% of total life)")
