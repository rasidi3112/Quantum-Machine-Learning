import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Data from 6 Runs
# Replace these values ​​with your actual experimental data.
data = {
    'VQC': {
        'Accuracy': [76.67, 88.33, 83.33, 86.67, 73.33, 78.33],
        'F1-Score': [0.748, 0.873, 0.828, 0.867, 0.733, 0.772],
        'ROC-AUC': [0.907, 0.957, 0.886, 0.922, 0.790, 0.850]
    },
    'QSVM': {
        'Accuracy': [50.0, 50.0, 50.0, 50.0, 50.0, 50.0],
        'F1-Score': [0.0, 0.667, 0.0, 0.0, 0.667, 0.0],
        'ROC-AUC': [0.459, 0.500, 0.500, 0.530, 0.500, 0.650]
    }
}

# Ensure the folder exists
os.makedirs('artifacts/figures', exist_ok=True)

# Ensure the folder exists
sns.set_theme()

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Accuracy
sns.boxplot(data=[data['VQC']['Accuracy'], data['QSVM']['Accuracy']], ax=axes[0])
axes[0].set_title('Test Accuracy')
axes[0].set_ylabel('Accuracy (%)')
axes[0].set_xticklabels(['VQC', 'QSVM'])
axes[0].grid(True, alpha=0.3)

# F1-Score
sns.boxplot(data=[data['VQC']['F1-Score'], data['QSVM']['F1-Score']], ax=axes[1])
axes[1].set_title('F1-Score')
axes[1].set_ylabel('F1-Score')
axes[1].set_xticklabels(['VQC', 'QSVM'])
axes[1].grid(True, alpha=0.3)

# ROC-AUC
sns.boxplot(data=[data['VQC']['ROC-AUC'], data['QSVM']['ROC-AUC']], ax=axes[2])
axes[2].set_title('ROC-AUC')
axes[2].set_ylabel('ROC-AUC')
axes[2].set_xticklabels(['VQC', 'QSVM'])
axes[2].grid(True, alpha=0.3)

plt.tight_layout()

# Save figure
plt.savefig('artifacts/figures/boxplot_metrics.png', dpi=300, bbox_inches='tight')
plt.show()

print("Boxplot saved to artifacts/figures/boxplot_metrics.png")
