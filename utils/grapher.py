import matplotlib.pyplot as plt

def draw(history):
    history.history.keys()
    plt.title('loss')

    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.xlabel("epoch")
    plt.ylabel('loss')
    plt.legend()

    plt.title('accuracy')

    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='test')
    plt.xlabel("epoch")
    plt.ylabel('accuracy')
    plt.legend();