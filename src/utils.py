def check_for_df(full_filename):
    """check to see if this folder has a csv file, if so, use it if it has the correct formatting"""
    if ".csv" in full_filename:
        df=pd.read_csv(full_filename)
        if "video" not in df.columns:
            print('error: there is a csv file without the proper structure in the video folder')
            

def get_video_descriptions(df, folder_or_avi):
    """ takes either an AVI or folder of AVIs as input 
    and retrieves the csv/dataframe of descriptions extracted
    from a frame of the video/the footer with temp/date/time"""
    if "AVI" in folder_or_avi[-4:] or "avi" in folder_or_avi[:-4]:
        print('an AVI, not a folder')
        df=text_parsing(df,video_to_text(full_path,[545,675,1280,720]))
    else:
        print("this is a folder")
        for file in os.listdir(folder_or_avi):
            if "AVI" in file:
                full_path = os.path.join(folder_or_avi,file)
                print("printing full_path in folder loop: {}".format(full_path))
                df=text_parsing(df,video_to_text(full_path,[545,675,1280,720]))
    return df


def video_to_text(video_path,rectangle_values):
    """this fx takes in a full video path
    and a list of rectangle values [minx,miny,maxx,maxy]
    it then uses this to draw a box around the text in a video still
    crop the still, and extract the text"""
    #unpack inputs
    [minx,miny,maxx,maxy]=rectangle_values
    print("video to text video_path = {}".format(video_path))
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
    print("text is {}".format(text))
    return text


def text_parsing(df,text):
    """this fx takes in a dataframe and text string, 
    and parses the information to fill the dataframe appropriately"""
    textlist = text.split()
    print("textlist is {}".format(textlist))
    # parse string into a data structure
    data={}
    data['video']=video_path.split('/')[-1]
    for substr_index in np.arange(len(textlist)):
        substr=textlist[substr_index]
        if "F" in substr:
            data['F']=get_temp(substr_index,substr,textlist)
        elif "C" in substr:
            data['C']=get_temp(substr_index,substr,textlist)
        elif "2021" in substr:
            data['date']=substr.split('/')[0]+substr.split('/')[1]+substr.split('/')[2]
        elif ":" in substr:
            data['time']=substr.split(':')[0]+substr.split(':')[1]+substr.split(':')[2]
    # sometimes the first degree measure, Celcius, gets missed, but we can calculate it from F
    if 'C' not in data:
        data['C']=str(round((int(data['F'])-32)*(5/9)))
    print(data)
    df=df.append(data,ignore_index=True)
    return df


def get_temp(index,substr,textlist):
    """this fx takes in an index and a substr responding to that index"""
    if substr.split('°')[0].isnumeric():
        print('if {}'.format(textlist[index]))
        num = substr.split('°')[0]
    elif textlist[substr_index-1].isnumeric():
        num = textlist[substr_index-1]
        print('elif {}'.format(textlist[index-1]))
    elif len(re.findall(r'\d+',textlist[substr_index-1]))>0:
        print('else before {}'.format(textlist[index-1]))
        num = num_from_text(textlist[index-1])
    else:
        temp_index=index
    print(num)
    return num


def num_from_text(string_input):
    temp = re.findall(r'\d+', string_input)
    num = list(map(int, temp))[0]
    return num