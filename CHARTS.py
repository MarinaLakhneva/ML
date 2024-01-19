import matplotlib.pyplot as plt


def plot_two_ys(y_pred, y_test):
    plt.figure(figsize=(15, 10))
    plt.plot(y_pred, label='y_pred')
    plt.plot(y_test, label='y_test')
    plt.legend(fontsize=20)
    plt.xlabel('Index of the observation', fontsize=20)
    plt.ylabel('Y', fontsize=20)
    plt.show()
