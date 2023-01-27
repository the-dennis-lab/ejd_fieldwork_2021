
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: ejdennis
purpose is to take in a video path or path to a folder of videos
and output a csv with the timestamp and weather info from that video
goal is to eventually also add to this csv the # of animals from DLC

input: a string, path to a video or folder full of videos
"""

import pandas as pd
import numpy as np
import glob, os, sys, math, warnings, argparse, time, cv2, pytesseract, re
import utils


if __name__ == "__main__":
    # deal with inputs
    try:
        fld=str(sys.argv[1])
        if fld[-4:].upper()==".AVI":
            fld = os.path.dirname(fld)
        if ~os.path.isdir(fld):
            print('the folder {} in the path does not exist, check your path'.format(fld))
    except:
        print("this function requires two inputs: the first must be ",
            "a string that leads to a folder of videos or a video file",
            "directly")
    try:
        csv_output=str(sys.argv[2])
        os.path.isdir(os.path.dirname(fld))
        if csv_output[-4:].lower() != ".csv":
            csv_output = csv_output+"_timestamps.csv"
    except:
        print("this function requires two inpus, the second input must be a file ",
        "name in a location that exists")
    df = pd.DataFrame(columns=['video','date','time','C','F'])
    utils.get_video_descriptions(df, fld, csv_output)
