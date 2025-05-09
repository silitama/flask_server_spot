def crop_plate(image, box):
    x1, y1, x2, y2 = map(int, box)
    return image[y1:y2, x1:x2]
