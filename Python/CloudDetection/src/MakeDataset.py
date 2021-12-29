import numpy as np
import cv2
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
import random
import copy


class Dataset(object):

    def __init__(self, satellite):
        # target satellite
        self.satellite = satellite

        self.image_data_format = None
        self.mask_data_format = None
        self.images = None
        self.masks = None
        self.n_data = None
        self.sample_weights = None

    def load_data(self, path, dataset_satellite, resize=True):
        """Loads data from specified path that needs to hold the folders 'masks' and 'images'. The dataset satellite specifies the dimensions and number of channels of the loaded data..

        Keyword arguments:
        resize: Resize the original data to be fittet to the width dimension of the satellite trained for.
        """
        # define paths
        image_path = path + "/images/"
        mask_path = path + "/masks/"

        # file names
        image_products = sorted(os.listdir(image_path))
        if image_products[0] == '.DS_Store':  # file in macOS systems, that should be ignored
            image_products = image_products[1:]
        mask_products = sorted(os.listdir(mask_path))
        if mask_products[0] == '.DS_Store':
            mask_products = mask_products[1:]
        self.n_data = len(image_products)

        #allocate
        images_original = np.zeros((self.n_data, dataset_satellite.img_height, dataset_satellite.img_width, dataset_satellite.n_bands), dtype=np.uint8)
        masks_original = np.zeros((self.n_data, dataset_satellite.img_height, dataset_satellite.img_width), dtype=np.uint8)

        # resizing
        resize_factor = self.satellite.img_width / dataset_satellite.img_width
        target_shape = (round(resize_factor*dataset_satellite.img_height), self.satellite.img_width)

        if resize:
            self.images = np.zeros((self.n_data, target_shape[0], target_shape[1], self.satellite.n_bands), dtype=np.uint8)
            self.masks = np.zeros((self.n_data, target_shape[0], target_shape[1]), dtype=np.uint8)

        else:
            self.images = np.zeros((self.n_data, dataset_satellite.img_height, dataset_satellite.img_width, dataset_satellite.n_bands), dtype=np.uint8)
            self.masks = np.zeros((self.n_data, dataset_satellite.img_height, dataset_satellite.img_width), dtype=np.uint8)

        # load spectral data into an array
        if image_products[1].find('npy') == -1:
            for i in range(1, self.n_data):
                images_original[i - 1, :, :, 0:self.satellite.n_bands] = cv2.imread(image_path + image_products[i - 1])
                images_original[i - 1, :, :, 0:3] = cv2.cvtColor(images_original[i - 1, :, :, :3], cv2.COLOR_BGR2RGB)


                if resize:
                    self.images[i - 1, :, :, 0:self.satellite.n_bands] = cv2.resize(images_original[i - 1, :, :, 0:self.satellite.n_bands],
                                                                                    dsize=target_shape)
                else:
                    self.images[i - 1] = images_original[i - 1]
        else:
            for i in range(1, self.n_data):
                images_original[i - 1] = np.load(image_path + image_products[i - 1], allow_pickle=True)


                if resize:
                    self.images[i - 1] = cv2.resize(images_original[i - 1, :, :, :],
                                                    dsize=target_shape)
                else:
                    self.images[i - 1] = images_original[i - 1]

        # load masks into an array
        if mask_products[0].find('npy') == -1:
            for i in range(1, self.n_data):
                masks_original[i - 1] = cv2.imread(mask_path + mask_products[i - 1])

                if resize:
                    self.masks[i - 1]= cv2.resize(masks_original[i - 1, :, :],
                                                         dsize=target_shape)
                else:
                    self.masks[i - 1] = masks_original[i - 1]
        else:
            for i in range(1, self.n_data):
                masks_original[i - 1] = np.load(mask_path + mask_products[i - 1], allow_pickle=True)

                if resize:
                    self.masks[i - 1] = cv2.resize(masks_original[i - 1, :, :],
                                                   dsize=target_shape)
                else:
                    self.masks[i - 1] = masks_original[i - 1]

    def patch_array(self, input_array, pixel_shift):
        """ Patch an array to fit the dimensions of the satellite trained for.

        Keyword arguments:
        pixel_shift: Shift between two patches
        """
        target_width = self.satellite.img_width
        target_height = self.satellite.img_height
        n_data_points, initial_height, initial_width = input_array.shape[0:3]
        n_horizontal_patches = 1 + round((initial_width - target_width) / pixel_shift)
        n_vertical_patches = 1 + round((initial_height - target_height) / pixel_shift)

        if initial_width >= target_width and initial_height >= target_height:
            # Case: mask array
            if np.ndim(input_array) == 3:
                # ideally include case that (initial_width - target_width)/pixel_shift is not round
                patched_array = np.zeros((n_data_points * n_horizontal_patches * n_vertical_patches,
                                         target_height,
                                         target_width),
                                         dtype=np.uint8)

                for n in range(n_data_points):
                    for h in range(n_vertical_patches):
                        for w in range(n_horizontal_patches):
                            i = n_horizontal_patches * n_vertical_patches * n + n_horizontal_patches * h + w
                            patched_array[i] = input_array[n, h * pixel_shift:target_height + h * pixel_shift, w * pixel_shift:target_width + w * pixel_shift]

            # Case: 4 dimensional array
            else:
                channels = input_array.shape[3]

                patched_array = np.zeros((n_data_points * n_horizontal_patches * n_vertical_patches,
                                         target_height,
                                         target_width,
                                         channels),
                                         dtype=np.uint8)

                for n in range(n_data_points):
                    for h in range(n_vertical_patches):
                        for w in range(n_horizontal_patches):
                            i = n_horizontal_patches * n_vertical_patches * n + n_horizontal_patches * h + w
                            patched_array[i] = input_array[n, h * pixel_shift:target_height + h * pixel_shift, w * pixel_shift:target_width + w * pixel_shift, :]

            return patched_array

        else:
            print('The dataset is not suitable for' +
                  self.satellite.satellite_name +
                  '. Use a different dataset or resize the input data.')

    def patch_dataset(self, pixel_shift=40):
        """ Patch the dataset to fit the dimensions of the satellite trained for.

                Keyword arguments:
                pixel_shift: Shift between two patches
        """
        if self.images.any():
            self.images = self.patch_array(self.images, pixel_shift=pixel_shift)
            self.masks = self.patch_array(self.masks, pixel_shift=pixel_shift)

            if self.sample_weights is not None:
                self.sample_weights = self.patch_array(self.sample_weights, pixel_shift=pixel_shift)

            self.n_data = len(self.images)
        else:
            print('There is no data loaded.')

    def transform_expand_dataset(self):
        """ Double the amount of data by randomly flipping each image over the x- and/or y-axis."""
        new_images = np.zeros((2 * len(self.images), self.satellite.img_height, self.satellite.img_width, self.images.shape[3]), dtype=np.uint8)
        new_masks = np.zeros((2 * len(self.images), self.satellite.img_height, self.satellite.img_width), dtype=np.uint8)
        n_data = len(self.images)

        for i in range(n_data):
            new_images[i] = self.images[i]
            new_masks[i] = self.masks[i]

        self.images = new_images
        self.masks = new_masks

        for i in range(n_data):
            r = random.random()

            if r < 0.33:
                self.images[n_data + i] = np.flip(self.images[i], 0)
                self.masks[n_data + i] = np.flip(self.masks[i], 0)
            if 0.33 < r < 0.66:
                self.images[n_data + i] = np.flip(self.images[i], 1)
                self.masks[n_data + i] = np.flip(self.masks[i], 1)
            else:
                self.images[n_data + i] = np.flip(self.images[i], 0)
                self.masks[n_data + i] = np.flip(self.masks[i], 0)

                self.images[n_data + i] = np.flip(new_images[n_data + i], 1)
                self.masks[n_data + i] = np.flip(new_masks[n_data + i], 1)

        self.n_data = len(self.images)

    def rotate_dataset(self, n_rotations=2):
        """ Expand the dataset by rotating each data product randomly between -30 and 30 degrees.

        Keyword arguments:
        n_rotations: Number of random rotations for each data product
        """
        # include sample weights!!!
        new_images = np.zeros(((1 + n_rotations) * len(self.images), self.satellite.img_height, self.satellite.img_width, self.images.shape[3]), dtype=np.uint8)
        new_masks = np.zeros(((1 + n_rotations) * len(self.images), self.satellite.img_height, self.satellite.img_width), dtype=np.uint8)

        n_data = len(self.images)

        for i in range(n_data):
            new_images[i] = self.images[i]
            new_masks[i] = self.masks[i]

        self.images = new_images
        self.masks = new_masks

        # calculate target size and center to rotate around
        (target_height, target_width) = (self.satellite.img_height, self.satellite.img_width)
        (cX, cY) = (target_width // 2, target_height // 2)

        for i in range(n_data):
            for n in range(1, n_rotations+1):
                angle = random.randint(-30, 30)
                M = cv2.getRotationMatrix2D((cX, cY), angle, 1.4)
                self.images[n*n_data + i] = cv2.warpAffine(self.images[i], M, (target_width, target_height))
                self.masks[n*n_data + i] = cv2.warpAffine(self.masks[i], M, (target_width, target_height))

        self.n_data = len(self.images)

    def produce_sample_weights(self, cloud_weight):
        """ Synthesise sample weights in order to weight the True Positive (TP) classifications.

        Keyword arguments:
        cloud_weight: Weight in comparison to True Negative classes, which are always 1
        """
        n_data, height, width = self.masks.shape
        self.sample_weights = np.zeros((n_data, height, width))

        for i in range(self.n_data):
            for h in range(height):
                for w in range(width):
                    if self.masks[i, h, w] == 0:
                        self.sample_weights[i, h, w] = 1
                    else:
                        self.sample_weights[i, h, w] = cloud_weight

    def show_image_mask(self, index, superposition=True):
        """ Show the indexed image alongside its mask."""
        height, width, channels = self.images[index].shape

        rcParams['figure.figsize'] = 13, 8
        fig, ax = plt.subplots(1, 2)
        colours = ['#62a6c2', 'gold']
        if np.unique(self.masks)[0] == 1:
            colours = ['gold']
        ax[0].imshow(cv2.cvtColor(self.images[index, :, :, :3], cv2.COLOR_BGR2RGB))
        ax[0].axis('off')
        if superposition:
            superposition = cv2.cvtColor(self.images[index, :, :, :3], cv2.COLOR_BGR2RGB)
            for h in range(height):
                for w in range(width):
                    if self.masks[index, h, w] == 1:
                        superposition[h, w, :] = (255, 110, 0)
            ax[1].imshow(superposition)
        else:
            ax[1].imshow(self.masks[index], cmap=matplotlib.colors.ListedColormap(colours))
        ax[1].axis('off')
        plt.show()

    def cloud_amount(self):
        """ Calculate the portion of clouds im the dataset rounded to two digits after the comma."""
        classes, counts = np.unique(self.masks, return_counts=True)
        percentage = counts[1] / sum(counts)
        percentage = round(percentage*10000)/100
        print('The percentage of cloud pixels is ' + str(percentage) + '%')





