#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import glob
import subprocess
import re
import numpy as np

if __name__ == '__main__':
    # Parse optios for downloading
    parser = argparse.ArgumentParser(description='Parse a folder and computes the mean and standard deviation.')
    # Optional argument
    parser.add_argument('folder', type=str, help='Specify the data directory.')

    args = parser.parse_args()

    f_list=glob.glob(args.folder+'*.txt')
    
    acc = []
    regex = re.compile(r"(?<=\bAverage Acc )[\w.]+")
    for f in f_list:
        line = subprocess.check_output(['tail', '-3', f])
        match = regex.search(line)
        acc.append(float(match.group()))
    acc = np.array(acc)
    print('Number of tests: {}\nMean: {}\nStandard Deviation {}'.format(len(acc), acc.mean(), acc.std()))
