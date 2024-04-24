import numpy as np
import collections
import heapq
import itertools
from comp.Utility import utility_
from pathlib import Path


class Lossless_Compression_Decompression:
    def __init__(self, b, threshold, Q=27, *args):

        self.image = np.random.randint(0, 256, size=(100, 100))
        self.M, self.N = self.image.shape
        self.b = b
        self.Q = Q
        self.threshold = threshold
        self.encoded_labels = None
        self.obj = utility_()
        self.img_path = args[0]
        self.my_string = []

    def huffman_encode(self, labels):
        # Perform Huffman coding on the labels
        counter = collections.Counter(labels)
        huffman_tree = [[weight, [label, ""]] for label, weight in
                        counter.items()]
        heapq.heapify(huffman_tree)
        while len(huffman_tree) > 1:
            lo = heapq.heappop(huffman_tree)
            hi = heapq.heappop(huffman_tree)
            for pair in lo[1:]:
                pair[1] = '0' + pair[1]
            for pair in hi[1:]:
                pair[1] = '1' + pair[1]
            heapq.heappush(huffman_tree, [lo[0] + hi[0]] + lo[1:] + hi[1:])
        huffman_encoding = dict(
            itertools.chain.from_iterable(huffman_tree[0][1:]))
        encoded_labels = [huffman_encoding[label] for label in labels]
        self.encoded_labels = encoded_labels
        return encoded_labels

    def partition(self, Ic, Q):
        blocks = []
        for i in range(0, self.M, Q):
            for j in range(0, self.N, Q):
                block = Ic[i:i + Q, j:j + Q]
                blocks.append(block)
        return blocks

    def find_pixel_rectangles(self, image):
        pixel_rectangles = []
        visited = np.zeros_like(image)  # Keep track of visited pixels

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if visited[i, j] == 0:
                    pixel_value = image[i, j]
                    rectangle = [(i, j)]  # Initialize a new rectangle
                    visited[i, j] = 1  # Mark pixel as visited

                    # Explore neighboring pixels
                    stack = [(i, j)]
                    while stack:
                        row, col = stack.pop()
                        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                            new_row, new_col = row + dx, col + dy
                            if 0 <= new_row < image.shape[0] and 0 <= new_col < \
                                    image.shape[1]:
                                if visited[new_row, new_col] == 0 and image[
                                    new_row, new_col] == pixel_value:
                                    rectangle.append((new_row, new_col))
                                    visited[new_row, new_col] = 1
                                    stack.append((new_row, new_col))

                    # Add rectangle to list if it's not empty
                    if rectangle:
                        pixel_rectangles.append(rectangle)

        return pixel_rectangles

    def smooth_block(self, blocks, threshold):
        labeled_blocks = []
        for block in blocks:
            pixel_rectangles = self.find_pixel_rectangles(block)
            epsilon = len(pixel_rectangles)
            if epsilon <= threshold:
                labeled_block = np.full_like(block, -1)
                labeled_blocks.append(labeled_block)
            else:
                labeled_blocks.append(block)
        return labeled_blocks

    def compress(self):
        # self.img_path = Path(r"Source\Lava.jpg.jpeg")
        self.my_string = self.obj.compress(str(self.img_path), 27)

        Ic = self.image // self.b
        Id = self.image - Ic * self.b

        blocks = self.partition(Ic, self.Q)
        labeled_blocks = self.smooth_block(blocks, self.threshold)

        encoded_labels = []
        for block in labeled_blocks:
            pixel_rectangles = self.find_pixel_rectangles(block)
            num_rectangles = len(pixel_rectangles)
            encoded_labels.extend([num_rectangles])

        return encoded_labels

    def huffman_decode(self):
        decoded_labels = self.huffman_decode(self.encoded_labels)

        reconstructed_image = np.zeros(self.image.shape)

        index = 0
        for i in range(self.M):
            for j in range(self.N):
                num_rectangles = decoded_labels[index]
                index += 1

                if num_rectangles == -1:
                    reconstructed_image[i, j] = decoded_labels[index]
                    index += 1
                else:
                    for _ in range(num_rectangles):
                        index += 1  # Skip rectangle information for now

        return reconstructed_image

    def decompress(self, encoded_labels):
        # Implement Huffman decoding to obtain difference sequence D
        self.obj.decompress(self.my_string[0], self.my_string[1],
                            self.my_string[2],
                            self.my_string[3], self.img_path)
        binary = []
        letter_binary = []
        bitstring = str(binary[2:])
        uncompressed_string = ""
        code = ""
        for digit in bitstring:
            code = code + digit
            pos = 0  # iterating and decoding
            for letter in letter_binary:
                if code == letter[1]:
                    uncompressed_string = uncompressed_string + \
                                          letter_binary[pos][
                                              0]
                    code = ""
                pos += 1
        return encoded_labels  # Placeholder for decoding

    def metric(self):
        cr = self.obj.metrics(self.threshold, self.Q)


def main():
    global img
    pics_path = ["france.tif", "frog.tif", "zelda.tif"]
    main_path = Path("Source")
    for path in pics_path:
        path_image = main_path / path
        image_path = str(Path(path_image))
        img = Lossless_Compression_Decompression(3, 21, 27, image_path)
        encoded_labels = img.compress()
        reconstructed_image = img.decompress(encoded_labels)
    img.metric()

if __name__ == "__main__":
    main()