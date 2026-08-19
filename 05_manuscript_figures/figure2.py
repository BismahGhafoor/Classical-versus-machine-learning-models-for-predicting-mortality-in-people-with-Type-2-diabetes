import matplotlib.pyplot as plt

# X-axis
models = ["Model 1", "Model 2", "Model 3", "Model 4"]
x = [1, 2, 3, 4]

# Data
female_allcause = [0.8975, 0.9090, 0.9109, 0.9172]
male_allcause   = [0.8551, 0.8704, 0.8730, 0.8808]

female_cvd = [0.8844, 0.8946, 0.8977, 0.8998]
male_cvd   = [0.8286, 0.8441, 0.8507, 0.8535]

female_cancer = [0.7980, 0.8137, 0.8164, 0.8188]
male_cancer   = [0.8015, 0.8142, 0.8160, 0.8183]

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharex=False)
fig.suptitle("Change in XGBoost AUC across predictor models", fontsize=16, y=1.03)

# -------- Panel A: All-cause mortality --------
ax = axes[0]
ax.plot(x, female_allcause, marker='o', linestyle='-', linewidth=2, label='Female')
ax.plot(x, male_allcause, marker='s', linestyle='--', linewidth=2, label='Male')
ax.set_title("A. All-cause mortality")
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=0)
ax.set_xlabel("Predictor model")
ax.set_ylabel("XGBoost AUC")
ax.set_ylim(0.84, 0.93)
ax.grid(True, alpha=0.3)

# -------- Panel B: CVD mortality --------
ax = axes[1]
ax.plot(x, female_cvd, marker='o', linestyle='-', linewidth=2, label='Female')
ax.plot(x, male_cvd, marker='s', linestyle='--', linewidth=2, label='Male')
ax.set_title("B. CVD mortality")
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=0)
ax.set_xlabel("Predictor model")
ax.set_ylabel("XGBoost AUC")
ax.set_ylim(0.81, 0.92)
ax.grid(True, alpha=0.3)

# -------- Panel C: Cancer mortality --------
ax = axes[2]
ax.plot(x, female_cancer, marker='o', linestyle='-', linewidth=2, label='Female')
ax.plot(x, male_cancer, marker='s', linestyle='--', linewidth=2, label='Male')
ax.set_title("C. Cancer mortality")
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=0)
ax.set_xlabel("Predictor model")
ax.set_ylabel("XGBoost AUC")
ax.set_ylim(0.79, 0.83)
ax.grid(True, alpha=0.3)

# Shared legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout(rect=[0, 0.07, 1, 0.95])

# Save outputs
plt.savefig("Figure2_XGBoost_AUC_across_models.png", dpi=300, bbox_inches="tight")
plt.savefig("Figure2_XGBoost_AUC_across_models.pdf", bbox_inches="tight")
plt.show()
