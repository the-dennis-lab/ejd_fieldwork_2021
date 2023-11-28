
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

def check_for_df(full_filename):
    """check to see if this folder has a csv file, if so, use it if it has the correct formatting"""
    if ".csv" in full_filename:
        df=pd.read_csv(full_filename)
        if "video" not in df.columns:
            print('error: there is a csv file without the proper structure in the video folder')


def get_video_descriptions(df, folder_or_avi, crop_vals):
    """ takes either an AVI or folder of AVIs as input
    and retrieves the csv/dataframe of descriptions extracted
    from a frame of the video/the footer with temp/date/time"""
    if "AVI" in folder_or_avi[-4:] or "avi" in folder_or_avi[:-4]:
        print('an AVI, not a folder')
        df=text_parsing(df,video_to_text(folder_or_avi,crop_vals),folder_or_avi)#[545,675,1280,720]))
    else:
        print("this is a folder")
        for file in os.listdir(folder_or_avi):
            if "AVI" in file:
                full_path = os.path.join(folder_or_avi,file)
                print("printing full_path in folder loop: {}".format(full_path))
                df=text_parsing(df,video_to_text(full_path,crop_vals),full_path)
    return df


def video_to_text(video_path,rectangle_values):
    """this fx takes in a full video path
    and a list of rectangle values [minx,miny,maxx,maxy]
    it then uses this to draw a box around the text in a video still
    crop the still, and extract the text"""
    #unpack inputs
    [minx,miny,maxx,maxy]=rectangle_values
    # get a still from the video
    img = cv2.VideoCapture(video_path)
    ret, frame = img.read()
    f2 = frame.copy()
    rect = cv2.rectangle(f2, (minx,miny), (maxx,maxy), (0, 255, 0), 2)
    # crop
    cropped=f2[miny:maxy,minx:maxx]

    # make grayscale
    gray = cv2.cvtColor(cropped,cv2.COLOR_BGR2GRAY)
    # threshold and binarize
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    # get text from image
    text = pytesseract.image_to_string(thresh)
    return text


def text_parsing(df,text, video_path):
    """this fx takes in a dataframe and text string,
    and parses the information to fill the dataframe appropriately"""
    textlist = text.split()
    # parse string into a data structure
    data={}
    data['video']=video_path.split('/')[-1]
    for substr_index in np.arange(len(textlist)):
        substr=textlist[substr_index]
        if "F" in substr:
            try:
                data['F']=get_temp(substr_index,substr,textlist)
            except:
                allFs=[substrs for substrs in textlist if "F" in substrs]
                data['F']=allFs
        elif "C" in substr:
            try:
                data['C']=get_temp(substr_index,substr,textlist)
            except:
                allCs=[substrs for substrs in textlist if "C" in substrs]
                date['C']=allCs
        elif "202" and "/" in substr:
            try:
                data['date']=substr.split('/')[0]+substr.split('/')[1]+substr.split('/')[2]
            except:
                alldates=[substrs for substrs in textlist if "/" in substrs]
                data['date']=alldates
        elif ":" in substr:
            try:
                data['time']=substr.split(':')[0]+substr.split(':')[1]+substr.split(':')[2]
            except:
                alltimes=[substrs for substrs in textlist if ":" in substrs]
                if len(alltimes)==2:
                    substr_1=alltimes[0]
                    substr_2=alltimes[1]
                    data['time']=substr_1.split(':')[0]+substr_1.split(':')[1]+substr_2.split(':')[0]+substr_2.split(':')[1]
    # sometimes the first degree measure, Celcius, gets missed, but we can calculate it from F
    if 'C' not in data:
        data['C']=str(round((int(data['F'])-32)*(5/9)))
    df=df.append(data,ignore_index=True)
    return df


def get_temp(substr_index,substr,textlist):
    """this fx takes in an index and a substr responding to that index"""
    if substr.split('°')[0].isnumeric():
        num = substr.split('°')[0]
    elif textlist[substr_index-1].isnumeric():
        num = textlist[substr_index-1]
    elif len(re.findall(r'\d+',textlist[substr_index-1]))>0:
        num = num_from_text(textlist[substr_index-1])
    else:
        temp_index=index
    return num


def num_from_text(string_input):
    temp = re.findall(r'\d+', string_input)
    num = list(map(int, temp))[0]
    return num




if __name__ == "__main__":
    # deal with inputs\
    default_crop = [820,1020,1920,1080]
    try:
        fld=str(sys.argv[1])
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
        print("this function requires two inputs, the second input must be a file ",
        "name in a location that exists")
    try:
        crop_vals=str(eval(sys.argv[3]))

        if isinstance(crop_vals,list):
            if len(crop_vals)==4:
                print("your crop values are {}".format(crop_vals))
            else:
                print("using default crop_values of {}, you entered this list {} but it was not of length 4".format(default_crop, crop_vals))
                crop_vals=default_crop
        else:
            crop_vals=default_crop
            print("using default crop values of {} because you did not enter a list".format(crop_vals))
    except:
        crop_vals=default_crop
        print("using default crop values of {} because you did not enter a list of four integers in [] as your third argument".format(crop_vals))
    df = pd.DataFrame(columns=['video','date','time','C','F'])
    new_df = get_video_descriptions(df, fld,crop_vals)
    new_df.to_csv(csv_output)
