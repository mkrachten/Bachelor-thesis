class Satellite(object):

    def __init__(self, img_height, img_width, n_bands, satellite_name):
        self.satellite_name = satellite_name

        if img_width >= img_height:
            self.img_width = img_width
            self.img_height = img_height
            self.n_bands = n_bands

        else:
            print('Image width has to be larger or same size as image height.')
