import numpy as np
import matplotlib.pyplot as plt
from DatasetHandler import read_idx
from sklearn.metrics import confusion_matrix
from Classifier import SVMClassifier
from RBM import RBM

def get_accuracy(labels, predictions):
    right = 0.0
    for true, predicted in zip(labels, predictions):
        if true == predicted: 
            right += 1.0
    return right / len(labels)

def get_wrong_classifications(labels, predictions):
    wrongs = []
    for index, labels in enumerate(zip(labels, predictions)):
        if labels[0] != labels[1]: 
            wrongs.append(index)
    return wrongs

def plot_wrong_classifications(images, predictions, misclassified_indexes, rows=5, columns=5):
    fig, axs = plt.subplots(rows, columns, sharex=True, sharey=True)
    for i in range(rows):
        for j in range(columns):
            axs[i, j].imshow(images[misclassified_indexes[i * rows + j]].reshape(28, 28))
            axs[i, j].set_title(predictions[misclassified_indexes[i * rows + j]])
    for ax in axs.flat:
        ax.label_outer()
    plt.show()

def show_random_classification(images, predictions, rows=5, columns=5):
    fig, axs = plt.subplots(rows, columns, sharex=True, sharey=True)
    random_indexes = np.random.random_integers(0, len(images)-1, (rows * columns,))
    for i in range(rows):
        for j in range(columns):
            shape = random_indexes.shape
            axs[i, j].imshow(images[random_indexes[i * rows + j]])
            axs[i, j].set_title(predictions[random_indexes[i * rows + j]])
    for ax in axs.flat:
        ax.label_outer()
    plt.show()


print('Loading RBM...')
rbm = RBM()
paths = ['.\\RBMs\\weights_1587597688.2092822.csv', '.\\RBMs\\input_bias_1587597688.2092822.csv', '.\\RBMs\\hidden_bias_1587597688.2092822.csv']
rbm.load_machine(paths)


rbm.load_data('.\\Dataset\\train-images')
train_data = rbm.generate_hidden_representation(rbm.samples)
train_labels = read_idx('train-labels')


rbm.load_data('.\\Dataset\\test-images')
test_data = rbm.generate_hidden_representation(rbm.samples)
test_labels = read_idx('test-labels')


print('Loading SVM classifier...')
classifier = SVMClassifier()
classifier.load_classifier(['.\\Classifiers\\coef_1587589776.9200084.csv', '.\\Classifiers\\intercept_1587589776.9218214.csv', '.\\Classifiers\\classes_1587589776.9238167.csv'])
predicted_train_labels = classifier.predict(train_data)
predicted_test_labels = classifier.predict(test_data)


train_conf_matrix = confusion_matrix(train_labels, predicted_train_labels)
test_conf_matrix = confusion_matrix(test_labels, predicted_test_labels)
print('Train confusion matrix')
print(train_conf_matrix)
print('Test confusion matrix')
print(test_conf_matrix)

train_accuracy = get_accuracy(train_labels, predicted_train_labels)
print(f'Train accuracy: {train_accuracy}')
test_accuracy = get_accuracy(test_labels, predicted_test_labels)
print(f'Test accuracy: {test_accuracy}')

images = read_idx('.\\Dataset\\test-images')
exits = ['q', 'quit', 'e', 'exit']
randoms = ['r', 'random']
wrongs = ['w', 'wrongs', 'm','misclassified']
while(True):
    print('Press enter to show random classifications')
    s = input()
    if s in exits:
        break
    elif s in randoms:
        show_random_classification(images, predicted_test_labels, 5, 5)
    elif s in wrongs:
        misclassified = get_wrong_classifications(test_labels, predicted_test_labels)
        plot_wrong_classifications(images, predicted_test_labels, misclassified)