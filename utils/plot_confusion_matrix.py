import matplotlib.pyplot as plt
import seaborn as sns
def plot_confusion_matrix(y_true, y_pred, activity_labels):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[activity_labels[i] for i in range(6)],
                yticklabels=[activity_labels[i] for i in range(6)])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
