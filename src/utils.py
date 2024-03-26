import matplotlib.pyplot as plt
import seaborn as sns
import csv

def plot_accuracy(history):
    plt.plot(history['train_acc'], label='train accuracy')
    plt.plot(history['val_acc'], label='validation accuracy')

    plt.title('Training history')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.ylim([0, 1]);

def plot_loss(history):
    plt.plot(history['train_loss'], label='train loss')
    plt.plot(history['val_loss'], label='validation loss')

    plt.title('Training history')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
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