import datetime
import matplotlib.pyplot as plt


def plot_metric(history, metric, path):
    train_metrics = history.history[metric]
    val_metrics = history.history['val_' + metric]

    epochs = range(1, len(train_metrics) + 1)

    plt.plot(epochs, train_metrics)
    plt.plot(epochs, val_metrics)
    plt.title('Training and validation ' + metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_" + metric, 'val_' + metric])
    plt.savefig(path + metric + " " + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), dpi=300)
    plt.show()
