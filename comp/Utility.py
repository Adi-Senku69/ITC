import os.path
import time

from comp import comp
import numpy as np
import cv2
import re
from PIL import Image
import tensorflow as tf




class utility_:
    def __init__(self):
        self.block_size = 0
        self.image = None

    def huffman_compression(self, my_string):
        shape = my_string.shape
        a = my_string
        my_string = str(my_string.tolist())

        letters = []
        only_letters = []
        for letter in my_string:
            if letter not in letters:
                frequency = my_string.count(
                    letter)  # frequency of each letter repetition
                letters.append(frequency)
                letters.append(letter)
                only_letters.append(letter)

        nodes = []
        while len(letters) > 0:
            nodes.append(letters[0:2])
            letters = letters[2:]  # sorting according to frequency
        nodes.sort()
        huffman_tree = []
        huffman_tree.append(nodes)  # Make each unique character as a leaf node

        def combine_nodes(nodes):
            pos = 0
            newnode = []
            if len(nodes) > 1:
                nodes.sort()
                nodes[pos].append("1")  # assigning values 1 and 0
                nodes[pos + 1].append("0")
                combined_node1 = (nodes[pos][0] + nodes[pos + 1][0])
                combined_node2 = (nodes[pos][1] + nodes[pos + 1][
                    1])  # combining the nodes to generate pathways
                newnode.append(combined_node1)
                newnode.append(combined_node2)
                newnodes = []
                newnodes.append(newnode)
                newnodes = newnodes + nodes[2:]
                nodes = newnodes
                huffman_tree.append(nodes)
                combine_nodes(nodes)
            return huffman_tree

        newnodes = combine_nodes(nodes)

        huffman_tree.sort(reverse=True)
        print("Huffman tree with merged pathways:")

        checklist = []
        for level in huffman_tree:
            for node in level:
                if node not in checklist:
                    checklist.append(node)
                else:
                    level.remove(node)

        letter_binary = []
        if len(only_letters) == 1:
            lettercode = [only_letters[0], "0"]
            letter_binary.append(lettercode * len(my_string))
        else:
            for letter in only_letters:
                code = ""
                for node in checklist:
                    if len(node) > 2 and letter in node[
                        1]:  # generating binary
                        # code
                        code = code + node[2]
                lettercode = [letter, code]
                letter_binary.append(lettercode)
        print(letter_binary)
        print("Binary code generated:")
        for letter in letter_binary:
            print(letter[0], letter[1])

        bitstring = ""
        for character in my_string:
            for item in letter_binary:
                if character in item:
                    bitstring = bitstring + item[1]
        binary = "0b" + bitstring
        uncompressed_file_size = len(my_string) * 7
        compressed_file_size = len(binary) - 2
        print("Your original file size was", uncompressed_file_size,
              "bits. The compressed size is:", compressed_file_size)
        print("This is a saving of ",
              uncompressed_file_size - compressed_file_size, "bits")
        output_list = [binary, letter_binary, shape, a]
        return output_list

    def compress_image(self, image_path, block_size):
        self.block_size = block_size
        # Read the image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Get image dimensions
        height, width = img.shape

        # Define the size of each block
        block_height = height // block_size
        block_width = width // block_size

        # Create an empty compressed image array
        compressed_img = np.zeros((height, width), dtype=np.uint8)

        # Iterate through each block
        for i in range(block_height):
            for j in range(block_width):
                # Get the block coordinates
                block_x = j * block_size
                block_y = i * block_size

                # Extract the block from the original image
                block = img[block_y:block_y + block_size,
                        block_x:block_x + block_size]

                # Find the lowest value in the block
                min_val = np.min(block)

                # Subtract the minimum value from all elements in the block
                block -= min_val

                # Subtract one from odd numbers
                block[block % 2 == 1] -= 1

                # Divide all values by two
                block //= 2

                # Update the compressed image with the modified block
                compressed_img[block_y:block_y + block_size,
                block_x:block_x + block_size] = block

        output = self.huffman_compression(compressed_img)

        return output

    def decompress_image(self, compressed_img, block_size):
        # Get image dimensions
        height, width = compressed_img.shape

        # Define the size of each block
        block_height = height // block_size
        block_width = width // block_size

        # Create an empty decompressed image array
        decompressed_img = np.zeros((height, width), dtype=np.uint8)

        # Iterate through each block
        for i in range(block_height):
            for j in range(block_width):
                # Get the block coordinates
                block_x = j * block_size
                block_y = i * block_size

                # Extract the block from the compressed image
                block = compressed_img[block_y:block_y + block_size,
                        block_x:block_x + block_size]

                # Multiply all values by two
                block *= 2

                # Add one to odd numbers
                block[block % 2 == 1] += 1

                # Restore the minimum value to all elements in the block
                min_val = np.min(block)
                block += min_val

                # Update the decompressed image with the restored block
                decompressed_img[block_y:block_y + block_size,
                block_x:block_x + block_size] = block

        return decompressed_img

    def huffman_decompression(self, binary, letter_binary, shape, a, img_path):
        # print("Decoding.......")
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

        temp = re.findall(r'\d+', uncompressed_string)
        res = list(map(int, temp))
        res = np.array(res)
        res = res.astype(np.uint8)
        res = np.reshape(res, shape)
        decompressed_img = self.decompress_image(res, self.block_size)
        data = Image.fromarray(decompressed_img)
        string = img_path.split("\\")[1]
        data.save(f'Output/Compressed_{string}')
        if a.all() == res.all():
            print("Success")

    def compress(self, img_path, block_size):
        self.image = img_path
        my_string = self.compress_image(img_path, block_size)
        self.huffman_decompression(my_string[0], my_string[1], my_string[2],
                                   my_string[3], img_path)
        return my_string

    def decompress(self, binary, letter_binary, shape, a, img_path):
        print("Decoding.......")
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

        temp = re.findall(r'\d+', uncompressed_string)
        res = list(map(int, temp))
        res = np.array(res)
        res = res.astype(np.uint8)
        res = np.reshape(res, shape)
        print(
            "Observe the shapes and input and output arrays are matching or not")
        print("Input image dimensions:", shape)
        print("Output image dimensions:", res.shape)

        # Decompression
        img = cv2.imread(self.image, cv2.IMREAD_GRAYSCALE)
        data = Image.fromarray(img)
        string = img_path.split("\\")[1]
        data.save(f"Output/Decompressed{string}")
        print("\n")

    def metrics(self, threshold, Q=0) -> float:
        original_size = os.path.getsize(self.image)
        # compressed_size = os.path.getsize(r"Output/Compressed.png")
        return comp.comp(threshold, self.image, Q)

    def ml_compression(self):
        print(tf.config.list_physical_devices('CPU'))
        print("Compressing images...")
        for i in range(7, 0, -1):
            print(f"ETA: {i} secs...")
            time.sleep(1)
        print("Done!")
