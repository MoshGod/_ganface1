#!/user/bin/env python    
#-*- coding:utf-8 -*- 

from matplotlib import pyplot as plt
import numpy as np

def plot_eps_acc(attack_method_str, epss, accuracy, savedir, is_save=False):

    plt.ylim(0, 100)
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 101, 10))
    plt.plot(epss, accuracy)
    plt.title(attack_method_str)
    plt.xlabel("eps")
    plt.ylabel("deception rate")

    if is_save:
        plt.savefig(savedir)
    plt.show()

def plot_image(img, label, name):

    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(img[i][0], cmap='gray', interpolation='none')
        plt.title("{}: {}".format(name, label[i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()


def plot_gender(model, test_x, test_y, size):
    predictions = model.predict(test_x)

    class_names = ["Female", "Male"]

    plt.figure(figsize=(12, 6))
    for i in range(min(9, len(test_y))):
        result = predictions[i]
        max_label = int(np.argmax(result))
        correct_label = int(np.argmax(test_y[i]))

        plt.subplot(3, 6, 2 * i + 1)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        img = test_x.reshape(test_x.shape[0], size, size)[i]
        plt.imshow(img)
        plt.xlabel("{} - prob:{:2.0f}%".format(class_names[max_label], 100 * np.max(result)))

        plt.subplot(3, 6, 2 * i + 2)
        plt.grid(False)
        plt.yticks([])
        plt.ylim([0, 1])
        bar = plt.bar(range(2), result)
        bar[max_label].set_color('red')
        bar[correct_label].set_color('green')

    plt.show()

