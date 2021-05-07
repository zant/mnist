import numpy as np
import gzip
import os

IMAGE_MODE = "i"
LABEL_MODE = "l"

MAGIC_NUMS = {}
MAGIC_NUMS[IMAGE_MODE] = 2051
MAGIC_NUMS[LABEL_MODE] = 2049

class UByte:
    isValid = False

    @staticmethod
    def show_image(image):
        for row in image:
            for pixel in row:
                symbol = "*" if pixel > 127 else "."
                print(symbol, end='')
            print("")

    def read_file(self):
        _, ext = os.path.splitext(self.path)

        if ext == '.gz':
            self.file = gzip.open(self.path, 'r')
        else:
            self.file = open(self.path, 'rb')


    def __init__(self, path, mode="i", read=10):
        self.path = path
        self.read = read
        self.mode = mode

        self.read_file()
        self.validate()

        self.parse_header()
        if mode == IMAGE_MODE:
            self.parse_images()
        else:
            self.parse_labels()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.file.close()
        
    def validate(self):
        if self.isValid:
            return 

        if self.read_bytes(4) == MAGIC_NUMS[self.mode]:
            self.isValid = True

    def read_bytes(self, size):
        return self.to_int(self.file.read(size))

    def parse_header(self):
        self.count = self.read_bytes(4)
        if self.mode == IMAGE_MODE: 
            self.rows = self.read_bytes(4)
            self.cols = self.read_bytes(4)

    def parse_images(self): 
        buf = self.file.read(self.read * self.cols * self.rows)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(self.read, self.rows, self.cols)
        self.data = data

    def parse_labels(self):
        buf = self.file.read(self.read)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(self.read)
        self.data = data

    def to_int(self, b):
        return int.from_bytes(b, "big")
