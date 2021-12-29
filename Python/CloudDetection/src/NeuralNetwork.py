import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from src import UNet
import matplotlib
from matplotlib import rcParams
import matplotlib.ticker as ticker
import cv2


class NeuralNetwork(object):

    def __init__(self, dataset):

        self.dataset = dataset
        self.n_classes = np.max(self.dataset.masks) + 1
        self.model = UNet.u_net_model(self.dataset.satellite.img_height, self.dataset.satellite.img_width,
                                      self.dataset.satellite.n_bands, self.n_classes)
        self.history = None
        self.prediction_images = None
        self.prediction_masks = None
        self.confusion_matrix = None
        self.accuracy = None
        self.recall = None
        self.precision = None
        self.F1 = None
        self.accuracy_vec = None
        self.recall_vec = None
        self.precision_vec = None

    def load_model(self, model_path, history_path):
        """Load a pre-trained model together with it's history."""
        self.model = keras.models.load_model(model_path)
        self.history = np.load(history_path, allow_pickle='TRUE').item()

    def train_model(self, save_path, suffix, n_epochs=25, validation_split=0.05, shuffle=False):
        """Train the model and save it and the emerging history in the save_path with a suiting suffix for specification.

        Keyword arguments:
        n_epochs: number of epochs trained
        validation_split: portion of dataset that is used for validation
        shuffle: shuffle the data while training

        """
        # convert masks to categorical format
        masks_cat = tf.keras.utils.to_categorical(self.dataset.masks, num_classes=self.n_classes)

        if self.dataset.sample_weights is not None:
            train = self.model.fit(self.dataset.images, masks_cat,
                                   sample_weight=self.dataset.sample_weights,
                                   batch_size=16,  # multipliable to 32
                                   verbose=1,
                                   shuffle=shuffle,
                                   epochs=n_epochs,
                                   validation_split=validation_split)
        else:
            train = self.model.fit(self.dataset.images, masks_cat,
                                   batch_size=16,  # multipliable to 32
                                   verbose=1,
                                   shuffle=shuffle,
                                   epochs=n_epochs,
                                   validation_split=validation_split)
        # save model and history
        self.history = train.history
        np.save(save_path + '/cloud_hist_' + suffix + '.npy', train.history)
        self.model.save(save_path + '/cloud_model_' + suffix + '.hdf5')

    def predict(self, images_array):
        """Predict two or more images from a (n,h,w,3) array."""
        self.prediction_images = images_array
        pre = self.model.predict(images_array)
        self.prediction_masks = np.argmax(pre, axis=-1)

    def draw_history(self):
        """Plot the history that is deposited in the class."""
        if self.history:
            accuracy = np.array(self.history['accuracy'])
            val_accuracy = np.array(self.history['val_accuracy'])
            cost = np.array(self.history['loss'])
            val_cost = np.array(self.history['val_loss'])
            epochs = np.arange(1, len(accuracy) + 1)

            plt.subplot(1, 2, 1)
            plt.plot(epochs, accuracy, label='training data')
            plt.plot(epochs, val_accuracy, label='validation data')
            plt.title('Mean accuracy of predictions')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(loc="lower right")
            axes = plt.gca()
            axes.yaxis.grid()
            axes.xaxis.set_major_locator(ticker.MultipleLocator(5))
            axes.xaxis.set_minor_locator(ticker.MultipleLocator(1))

            plt.subplot(1, 2, 2)
            plt.plot(epochs, cost, label='training data')
            plt.plot(epochs, val_cost, label='validation data')
            plt.title('Mean cost of predictions')
            plt.ylabel('cost')
            plt.xlabel('epoch')
            plt.legend(loc="upper right")
            axes = plt.gca()
            axes.yaxis.grid()
            axes.xaxis.set_major_locator(ticker.MultipleLocator(5))
            axes.xaxis.set_minor_locator(ticker.MultipleLocator(1))
            plt.show()

        else:
            print('There is no history to be plotted.')

    def evaluate_prediction(self, validation_masks):
        """Evaluate the prediction with the help of the suiting masks.
        Saves Confusion Matrix, Recall, Precision, and F-score in the class.
        """
        if self.prediction_masks is not None:

            if self.prediction_masks.shape == validation_masks.shape:
                n_masks, height, width = self.prediction_masks.shape
                self.confusion_matrix = np.zeros((2, 2))
                TP = np.zeros(n_masks)  # True Positive (Cloud)
                FP = np.zeros(n_masks)  # False Positive (False Cloud)
                FN = np.zeros(n_masks)  # False Negative (False Clear)
                TN = np.zeros(n_masks)  # True Negative (Clear)
                self.accuracy_vec = np.zeros(n_masks)
                self.recall_vec = np.zeros(n_masks)
                self.precision_vec = np.zeros(n_masks)
                for i in range(n_masks):
                    for h in range(height):
                        for w in range(width):
                            if validation_masks[i, h, w] == 1:

                                if self.prediction_masks[i, h, w] == validation_masks[i, h, w]:
                                    TP[i] += 1
                                else:
                                    FN[i] += 1
                            else:

                                if self.prediction_masks[i, h, w] == validation_masks[i, h, w]:
                                    TN[i] += 1
                                else:
                                    FP[i] += 1
                    self.accuracy_vec[i] = (TP[i] + TN[i]) / (TP[i] + TN[i] + FP[i] + FN[i])
                    self.recall_vec[i] = TP[i] / (TP[i] + FN[i])
                    self.precision_vec[i] = TP[i] / (TP[i] + FP[i])
                if (sum(TP) + sum(FN)) == 0:
                    self.confusion_matrix[0, 0] = 0
                    self.confusion_matrix[1, 0] = 0
                else:
                    self.confusion_matrix[0, 0] = sum(TP) / (sum(TP) + sum(FN))
                    self.confusion_matrix[1, 0] = sum(FN) / (sum(TP) + sum(FN))
                if (sum(TP) + sum(FN)) == 0:
                    self.confusion_matrix[0, 0] = 0
                    self.confusion_matrix[1, 0] = 0
                else:
                    self.confusion_matrix[0, 1] = sum(FP) / (sum(TN) + sum(FP))
                    self.confusion_matrix[1, 1] = sum(TN) / (sum(TN) + sum(FP))

                self.accuracy = ((sum(TP) + sum(TN)) / (sum(TP) + sum(TN) + sum(FP) + sum(FN)))
                self.recall = self.confusion_matrix[0, 0]
                self.precision = sum(TP) / (sum(TP) + sum(FP))
                self.F1 = 2/(1/self.precision + 1/self.recall)



            else:
                'The validation masks have the wrong shape.'
        else:
            'There is no prediction, that can be evaluated.'

    def show_prediction(self, index, superposition=False):
        """Shows indexed prediction alongside the original image.

        Keyword arguments:
        superposition: The True Positive (TP) classifications are drawn on the original image"""
        height, width, channels = self.prediction_images[index].shape

        rcParams['figure.figsize'] = 13, 8
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(cv2.cvtColor(self.prediction_images[index, :, :, :3], cv2.COLOR_BGR2RGB))
        #ax[0].imshow(self.prediction_images[index, :, :, :3])
        ax[0].axis('off')
        if superposition:
            prediction_superposition = cv2.cvtColor(self.prediction_images[index, :, :, :3], cv2.COLOR_BGR2RGB)
            #prediction_superposition = self.prediction_images[index, :, :, :3]
            for h in range(height):
                for w in range(width):
                    if self.prediction_masks[index, h, w] == 1:
                        prediction_superposition[h, w, :] = (255, 110, 0)
            ax[1].imshow(prediction_superposition)
        else:
            colours = ['#62a6c2', 'gold']
            if np.unique(self.prediction_masks)[0] == 1:
                colours = ['gold']
            ax[1].imshow(self.prediction_masks[index], cmap=matplotlib.colors.ListedColormap(colours))
        ax[1].axis('off')
        plt.show()
