import matplotlib.pyplot as plt

# Function to plot training and validation metrics
def plot_epoch_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])

# Plot the training and validation metrics
def plot_nn_training(history):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plot_epoch_graphs(history, 'accuracy')
    plt.ylim(None, 1)
    plt.subplot(1, 2, 2)
    plot_epoch_graphs(history, 'loss')
    plt.ylim(0, None)
    plt.show()