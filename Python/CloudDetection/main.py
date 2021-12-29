from src import MakeDataset, Satellite, NeuralNetwork, TIR
import datetime
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import tifffile as tiff
import tensorflow as tf


def main():
    techno_sat = Satellite.Satellite(480, 640, 3, 'TechnoSat')
    landsat_8 = Satellite.Satellite(1000, 1000, 4, 'Landsat 8')
    landsat_8_input = Satellite.Satellite(1000, 1000, 4, 'Landsat 8')
    dataset = MakeDataset.Dataset(techno_sat)
    sentinel_2 = Satellite.Satellite(1022, 1022, 3, 'Sentinel 2')

    dataset.load_data('data/L8_SPARCS', landsat_8, resize=True)
    #dataset.show_image_mask(4,superposition=False)
    #dataset.show_image_mask(51, superposition=True)
    dataset.patch_dataset(pixel_shift=160)
    #dataset.show_image_mask(24, superposition=True)
    network = NeuralNetwork.NeuralNetwork(dataset)
    network.load_model('models/cloud_model_2021-11-15.hdf5', 'histories/cloud_hist_2021-11-15.npy')
    #network.draw_history()

    network.predict(dataset.images)
    network.evaluate_prediction(dataset.masks)
    print(network.accuracy)
    print(network.F1)
    #network.show_prediction(51, superposition=True)



if __name__ == '__main__':

    main()


