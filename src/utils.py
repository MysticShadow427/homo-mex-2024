import matplotlib.pyplot as plt
import seaborn as sns
import csv

def plot_accuracy_loss(history):

    train_acc = history['train_acc']
    train_loss = history['train_loss']
    val_acc = history['val_acc']
    val_loss = history['val_loss']

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_acc) + 1), train_acc, label='Train Accuracy', marker='o')
    plt.plot(range(1, len(val_acc) + 1), val_acc, label='Validation Accuracy', marker='o')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_loss) + 1), train_loss, label='Train Loss', marker='o')
    plt.plot(range(1, len(val_loss) + 1), val_loss, label='Validation Loss', marker='o')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()
    
def save_training_history(history,path):
    with open(path, 'w', newline='') as csvfile:
        fieldnames = ['epoch', 'train_acc', 'train_loss', 'val_acc', 'val_loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for epoch in range(len(history['train_acc'])):
            writer.writerow({'epoch': epoch,
                            'train_acc': history['train_acc'][epoch],
                            'train_loss': history['train_loss'][epoch],
                            'val_acc': history['val_acc'][epoch],
                            'val_loss': history['val_loss'][epoch]})

    print("History saved to", path)

