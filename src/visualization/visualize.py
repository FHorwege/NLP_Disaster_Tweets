import matplotlib.pyplot as plt

# Function to plot training and validation metrics
def plot_epoch_graphs(history):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plot_line(history, 'accuracy')
    plt.ylim(None, 1)
    plt.subplot(1, 2, 2)
    plot_line(history, 'loss')
    plt.ylim(0, None)
    plt.show()
    
def plot_line(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])