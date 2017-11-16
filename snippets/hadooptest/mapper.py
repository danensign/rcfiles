#!/usr/local/bin/python3

import io
import sys

# decode
input_stream = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

for line in input_stream:
    line = line.strip()
    words = line.split()
    for word in words:
        print('{}\t{}'.format(word,1))
