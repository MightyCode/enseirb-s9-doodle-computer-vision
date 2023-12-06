import cv2
import numpy as np

class ImageCreation:
    @staticmethod
    def get_bounds(data, factor=10):
        """Return bounds of data."""
        min_x = None
        max_x = None
        min_y = None
        max_y = None

        abs_x = 0
        abs_y = 0
        for i in range(len(data)):
            x = float(data[i, 0]) / factor
            y = float(data[i, 1]) / factor
            abs_x += x
            abs_y += y

            if min_x is None:
                min_x = abs_x
            if max_x is None:
                max_x = abs_x
            if min_y is None:
                min_y = abs_y
            if max_y is None:
                max_y = abs_y

            min_x = min(min_x, abs_x)
            min_y = min(min_y, abs_y)
            max_x = max(max_x, abs_x)
            max_y = max(max_y, abs_y)

        return np.array([min_x, max_x, min_y, max_y])


    #   data is an array of 3-tuples, consisting of x-offset, y-offset, and a binary
    #       variable which is 1 if the pen is lifted between this position and
    #       the next, and 0 otherwise.
    # Return np array, center the drawing
    def createImage(data, image_size, stroke_size=3, factor=1):
        """Convert a sketch into a 2D array."""

        bounds = ImageCreation.get_bounds(data, factor)
        
        size = [bounds[1] - bounds[0], bounds[3] - bounds[2]]

        # start of drawing in zero, zero

        image = np.zeros(image_size, np.uint8)

        factors = [image_size[0] / size[0] / factor, image_size[1] / size[1] / factor]

        if factors[0] < factors[1]:
            factors[1] = factors[0]
        else:
            factors[0] = factors[1]

        remaining_space = [image_size[0] - size[0] * factors[0], image_size[1] - size[1] * factors[1]]
        
        pen_x, pen_y = -bounds[0] * factors[0] + remaining_space[0] / 2, -bounds[2] * factors[1] + remaining_space[1] / 2

        for i in range(len(data)):
            d_x = float(data[i, 0]) / factor * factors[0]
            d_y = float(data[i, 1]) / factor * factors[1]

            next_x = pen_x + d_x
            next_y = pen_y + d_y

            if i > 0 and data[i - 1, 2] == 0:
                cv2.line(image, (int(pen_x), int(pen_y)), (int(next_x), int(next_y)), 255, stroke_size)

            pen_x = next_x
            pen_y = next_y
        
        # Apply Gaussian Blur to blur the strokes
        image = cv2.GaussianBlur(image, (3, 3), 0)

        return image
    