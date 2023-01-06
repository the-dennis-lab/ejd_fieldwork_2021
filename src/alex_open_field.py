
"""alex_open_field.py:
REQUIRED INPUTS:
    1. a folder or file

OPTIONAL INPUTS:
    2. a trained .pkl file or tries to find one in ../data.
    3. a csv that has 'trim times' for videos that do not start at the 'correct' moment (camera started early)

HARDCODED:
    a few things are currently hardcoded for convenience.
    - If using a file to subset the dataframes by start/end time,
    then we assume you're using filenames like VID_20210925_134723 for your videos, that have 18 long strings. Else,
    search for 'filename.split('/')[-1][0:18]' and change 18 to a number of your choosing.
    - FPS is 29.93s, change at top of main

OUTPUTS:
several plots and csv files in the ../data/results folder and makes that folder if it does not yet exist"""

__author__ = "ejd"
__credits__ = ["ejd"]
__maintainer__ = "ejd"
__email__ = "dennise@hhmi.org"
__license__ = "MIT"
__status__ = "Development"

import numpy as np
import pandas as pd
import glob, os, csv, sys, cv2, math, itertools, joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.io import arff
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier


def reformat_df(df,likelihood_val,max_px_value,frame_limit_val,seconds,fps):
    new_df=pd.DataFrame()
    cols = np.unique([col[0] for col in df.columns])
    for col in cols:
        if "bodyparts" not in col:
            part = df[col]
            for idx in np.arange(0,len(part.x)):
                if part.likelihood[idx] < likelihood_val:
                    part.x[idx] = np.nan
                    part.y[idx] = np.nan
            if "box" not in col and "opening" not in col and "bodyparts" not in col:
                for idx in np.arange(0,len(part.x)):
                    second_idx = np.max([0,idx-1])
                    if abs(part.y[idx]-part.y[second_idx]) > max_px_value:
                        part.y[idx] = np.nan
                    elif abs(part.x[idx]-part.x[second_idx]) > max_px_value:
                        part.x[idx] = np.nan
                part.x = part.x.interpolate(limit=frame_limit_val)
                part.y = part.y.interpolate(limit=frame_limit_val)
                new_part = pd.concat({col:part},axis=1)
                if col == cols[1]:
                    new_df=new_part
                else:
                    new_df = pd.concat([new_df,new_part],axis=1)
            else:
                part.x = part.x.interpolate()
                part.x=part.x.rolling(seconds*fps,min_periods=1).median()
                part.y = part.y.interpolate()
                part.y=part.y.rolling(seconds*fps,min_periods=1).median()
                new_location = pd.concat({col:part},axis=1)
                new_df=pd.concat([new_df,new_location],axis=1)

    return new_df

def align_df(df):
    new_df=pd.DataFrame()
    # for each index, get the box coords, find transformation matrix, then apply it to all other points
    s1=[0,500]
    s2=[500,500]
    s3=[0,0]
    s4=[500,0]
    for idx in np.arange(0,len(df)):
        # get box points
        p1=[df.box_bl.x[idx],df.box_bl.y[idx]] #0,500
        p2=[df.box_br.x[idx],df.box_br.y[idx]] #500,500
        p3=[df.box_tl.x[idx],df.box_tl.y[idx]] #0,0
        p4=[df.box_tr.x[idx],df.box_tr.y[idx]] #500,0
        # get transform matrix
        M = cv2.getPerspectiveTransform(np.float32([p1,p2,p3,p4]),np.float32([s1,s2,s3,s4]))
        # apply transform matrix to other points
        sub_df=df[df.index==idx].copy()
        for i in np.arange(0,len(sub_df.columns),3):
            p1_0=sub_df.iloc[0,i]
            p1_1=sub_df.iloc[0,i+1]
            if np.isnan(p1_0):
                p1_0_new=np.nan
                p1_1_new=np.nan
            else:
                # set points
                pts = np.array([[[p1_0,p1_1]]],dtype='float32')
                # use transform matrix to un-warp the points
                [p1_0_new,p1_1_new]=cv2.perspectiveTransform(pts,M)[0][0]
            sub_df.iloc[0,i]=p1_0_new
            sub_df.iloc[0,i+1]=p1_1_new
        new_df=pd.concat([new_df,sub_df],axis=0)
    return new_df

if __name__ == "__main__":

    os.chdir(os.path.dirname(__file__))

    FPS=29.93
    frames_with_nose=[]
    frames_with_head=[]
    frames_with_body=[]
    total_frames=[]
    frames_within_50mm=[]
    fraction_frames_within_50mm=[]
    fraction_LIGHT_frames_within_50mm=[]
    fraction_LIGHT_frames_in_center=[]
    first_frame_in_center=[]
    num_entrances=[]

    if not os.path.isdir('../data/results'):
        os.mkdir('../data/results')

    print('===== printing sysargvs')
    print(sys.argv)
    print('===== end ')
    # arg parser
    file_or_folder=sys.argv[1]
    if os.path.isdir(file_or_folder):
        folder_of_files = file_or_folder
        folder_list = os.listdir(file_or_folder)
        full_path_list = [os.path.join(file_or_folder,file) for file in folder_list]
    elif os.path.isfile(file_or_folder):
        full_path_list = [file_or_folder]
    else:
        print('after alex_open_field, please provide a file or folder. You typed {} and this does not exist'.format(file_or_folder))

    if len(sys.argv) > 2 and os.path.isfile(sys.argv[2]):
        trained_file = sys.argv[2]
    elif os.path.isfile('../data/20221219_RF_jumps.pkl'):
        trained_file = '../data/20221219_RF_jumps.pkl'
    else:
        print('there is no trained .pb file in ../data and you did not provide a path to one. Please provide a path to one')

    # we may want to subset our DLC-run videos by some csv file
    to_subset=0
    if len(sys.argv) > 3:
        try:
            times = pd.read_csv(sys.argv[3])
            to_subset=1
        except:
            print('tried to use file {} to subset other files by time but failed to find the file'.format(sys.argv[3]))

        try:
            video_filenames=times.File_name
            video_start_frames=times.Start_time_in_seconds*FPS
            video_end_frames = times.End_time_in_seconds*FPS
            to_subset=1
        except:
            print('tried to use file {} to subset other files by time but failed because {} does not have columns called File_name, Start_time_in_seconds, and End_time_in_seconds')
    else: # look for files in the provided folder and in data
        if os.path.isfile('../data/start_times_videos.csv'):
            times = pd.read_csv('../data/start_times_videos.csv')
            to_subset=1
        elif os.path.isfile(os.path.join(file_or_folder,'start_times_videos.csv')):
            times = pd.read_csv(os.path.join(file_or_folder,'start_times_videos.csv'))
            to_subset=1
        else:
            print('assuming files have already been subsetted by start time, moving on without modifying them. If in error, retry and add third input')

    # load pickle trained file
    clf = joblib.load(trained_file)

    print('===== loaded everything successfully, now analyzing video csv(s)')

    for filename in full_path_list:
        df = pd.read_csv(filename,header=[1,2])
        new_df=reformat_df(df,0.9,1000,15,3,30)
        sub_file_name = filename.split('/')[-1][0:18]
        if to_subset:
            try:
                times_vals=times[times.File_name==sub_file_name]
                start_val=round(list(times_vals.Start_time_in_seconds)[0]*FPS)
                end_val=round(list(times_vals.Start_time_in_seconds)[0]*FPS)
                new_df=new_df[new_df.index<end_val]
                new_df=new_df[new_df.index>start_val]
                new_df=new_df.reset_index()
                new_df=new_df.drop(columns=['index'])
            except:
                print('failed to subset filename, skipping subsetting')
        new_df.to_csv('../data/results/'+sub_file_name+"_adj.csv")
        print('===== saved out {} adjusted csv, plotting then aligning next'.format(sub_file_name))

        # plot new corners
        plt.scatter(new_df.box_bl['x'],new_df.box_bl['y'],c='r') #0,500mm
        plt.scatter(new_df.box_br['x'],new_df.box_br['y'],c='pink') #500,500mm
        plt.scatter(new_df.box_tl['x'],new_df.box_tl['y'],c='b') #0,0mm
        plt.scatter(new_df.box_tr['x'],new_df.box_tr['y'],c='cyan') #500,0mm
        plt.savefig('../data/results/'+sub_file_name+"_adj_points.png")

        # affine transform to correct for sliding FOV
        aligned_df=align_df(new_df)
        aligned_df.to_csv('../data/results/'+sub_file_name++"_aligned.csv")
        print('===== saved out {} aligned csv, plotting then getting summary data'.format(sub_file_name))

        # plot newly aligned points
        plt.scatter(aligned_df.nose['x'],aligned_df.nose['y'],c='k',s=1,alpha=0.5)
        plt.scatter(aligned_df.box_bl['x'],aligned_df.box_bl['y'],c='r')
        plt.scatter(aligned_df.box_br['x'],aligned_df.box_br['y'],c='pink')
        plt.scatter(aligned_df.box_tl['x'],aligned_df.box_tl['y'],c='b')
        plt.scatter(aligned_df.box_tr['x'],aligned_df.box_tr['y'],c='cyan')
        plt.savefig('../data/results/'+sub_file_name+"_all_points_with_corners.png")

        # GET SUMMARY DATA
        # TODO SAVE OUT AS CSV FFTER LOOP
        nose_vals = ~np.isnan(aligned_df.nose['x'])
        ear_l_vals = ~np.isnan(aligned_df.ear_left['x'])
        ear_r_vals = ~np.isnan(aligned_df.ear_right['x'])
        tail_vals = ~np.isnan(aligned_df.tail_base['x'])

        # can just use adj px values > 450 and < 50 (further vals are outside walls, but include jumps and rears which we do want to include)
        nose_df=aligned_df.nose
        #nose_df=nose_df[~np.isnan(nose_df.x)].reset_index()
        nose_df['within50mm']=0
        for idx in nose_df.index:
            xval=nose_df.x[idx]
            yval=nose_df.y[idx]
            if xval < 50 or xval > 450 or yval < 50 or yval > 450:
                nose_df.iloc[idx,3]=1

        summed_val=np.sum(nose_df.within50mm)
        frames_with_nose.append(np.sum(nose_vals))
        frames_with_head.append(np.sum(nose_vals*ear_l_vals*ear_r_vals))
        frames_with_body.append(np.sum(nose_vals*ear_l_vals*ear_r_vals*tail_vals))
        total_frames = len(nose_vals)
        fraction_frames_within_50mm.append(summed_val/len(nose_df))
        fraction_LIGHT_frames_in_center.append((frames_with_nose-summed_val)/frames_with_nose)
        fraction_LIGHT_frames_within_50mm.append(summed_val/frames_with_nose)
        sub_nose_df=nose_df[~np.isnan(nose_df.x)]
        sub_nose_df=sub_nose_df[sub_nose_df.within50mm<1]
        first_frame_in_center.append(sub_nose_df.index[0])

        # add 2 cols to df: one for if the animal was in box or not, and one for the "bout number" which gives us # of entrances
        aligned_df['in_box']=float(0)
        aligned_df['light_bout_num']=float(0)
        bout_num=0
        for idx in aligned_df.index:
            if np.isnan(aligned_df.nose.x[idx]):
                aligned_df.iloc[idx,-2]=0
                aligned_df.iloc[idx,-1]=0
            elif aligned_df.iloc[idx-1,-2]==0:
                bout_num+=1
                aligned_df.iloc[idx,-2]=1
                aligned_df.iloc[idx,-1]=bout_num
            else:
                aligned_df.iloc[idx,-2]=1
                aligned_df.iloc[idx,-1]=bout_num
        num_entrances.append(np.max(aligned_df.light_bout_num))

        # plot inside and outside the walls
        plt.scatter(nose_df.x,nose_df.y,s=1,c='k')
        plt.scatter(nose_df.x[nose_df.x<50],nose_df.y[nose_df.x<50],c='pink',s=1)
        plt.scatter(nose_df.x[nose_df.x>450],nose_df.y[nose_df.x>450],c='pink',s=1)
        plt.scatter(nose_df.x[nose_df.y<50],nose_df.y[nose_df.y<50],c='pink',s=1)
        plt.scatter(nose_df.x[nose_df.y>450],nose_df.y[nose_df.y>450],c='pink',s=1)
        plt.scatter(nose_df.x[nose_df.x<0],nose_df.y[nose_df.x<0],c='r',s=1)
        plt.scatter(nose_df.x[nose_df.x>500],nose_df.y[nose_df.x>500],c='r',s=1)
        plt.scatter(nose_df.x[nose_df.y<0],nose_df.y[nose_df.y<0],c='r',s=1)
        plt.scatter(nose_df.x[nose_df.y>500],nose_df.y[nose_df.y>500],c='r',s=1)
        plt.savefig('../data/results/'+sub_file_name+"_all-points_color_within50mm.png")

        print('===== done collecting summary data for {}, now predicting jumps'.format(sub_file_name))

        # add distances between animal's body points
        aligned_df['dists','nose_ear_right']=float(0)
        aligned_df['dists','nose_ear_left']=float(0)
        aligned_df['dists','nose_paw_left_back']=float(0)
        aligned_df['dists','nose_paw_left_front']=float(0)
        aligned_df['dists','nose_paw_right_back']=float(0)
        aligned_df['dists','nose_paw_right_front']=float(0)
        aligned_df['dists','nose_tail_tip']=float(0)
        aligned_df['dists','nose_tail_base']=float(0)
        aligned_df['speed']=float(0)

        for idx in aligned_df.index:
            if idx%1000==0:
                print("on idx {} of {}".format(idx,aligned_df.index[-1]))
            rel_idx = idx-aligned_df.index[0]
            if rel_idx > 0:
                speed = math.dist([aligned_df.nose.x[idx],aligned_df.nose.y[idx]],[aligned_df.nose.x[idx-1],aligned_df.nose.y[idx-1]])
            else:
                speed=0
            aligned_df.iloc[rel_idx,-1]=speed
            aligned_df.iloc[rel_idx,-2]=math.dist([aligned_df.nose.x[idx],aligned_df.nose.y[idx]],[aligned_df.tail_base.x[idx],aligned_df.tail_base.y[idx]])
            aligned_df.iloc[rel_idx,-3]=math.dist([aligned_df.nose.x[idx],aligned_df.nose.y[idx]],[aligned_df.tail_tip.x[idx],aligned_df.tail_tip.y[idx]])
            aligned_df.iloc[rel_idx,-4]=math.dist([aligned_df.nose.x[idx],aligned_df.nose.y[idx]],[aligned_df.paw_right_front.x[idx],aligned_df.paw_right_front.y[idx]])
            aligned_df.iloc[rel_idx,-5]=math.dist([aligned_df.nose.x[idx],aligned_df.nose.y[idx]],[aligned_df.paw_right_back.x[idx],aligned_df.paw_right_back.y[idx]])
            aligned_df.iloc[rel_idx,-6]=math.dist([aligned_df.nose.x[idx],aligned_df.nose.y[idx]],[aligned_df.paw_left_front.x[idx],aligned_df.paw_left_front.y[idx]])
            aligned_df.iloc[rel_idx,-7]=math.dist([aligned_df.nose.x[idx],aligned_df.nose.y[idx]],[aligned_df.paw_left_back.x[idx],aligned_df.paw_left_back.y[idx]])
            aligned_df.iloc[rel_idx,-8]=math.dist([aligned_df.nose.x[idx],aligned_df.nose.y[idx]],[aligned_df.ear_left.x[idx],aligned_df.ear_left.y[idx]])
            aligned_df.iloc[rel_idx,-9]=math.dist([aligned_df.nose.x[idx],aligned_df.nose.y[idx]],[aligned_df.ear_right.x[idx],aligned_df.ear_right.y[idx]])
        aligned_df.to_csv('../data/results/'+sub_file_name+"_aligned_dists.csv")

        # select these columns for predicting jumps (used for training)
        df_for_prediction=pd.concat([aligned_df['nose']['x'],aligned_df['nose','y'],aligned_df['nose']['likelihood'],aligned_df['tail_base']['likelihood'],aligned_df['tail_tip']['likelihood'],aligned_df['new_speed']],axis=1)
        df_for_prediction.columns=['nose_x','nose_y','nose_likelihood','tail_base_likelihood','tail_tip_likelihood','speed']

        # rescale only the large value columns, so all columns are between 0, 1
        for col in ['nose_x','nose_y','speed']:
                maxval=np.nanmax(df_for_prediction[col])
                df_for_prediction[col]=df_for_prediction[col]/np.nanmax(df_for_prediction[col])

        preds_to_plot=clf.predict(df_for_prediction.fillna(0))

        plt.scatter(aligned_df.nose.x,aligned_df.nose.y,c=preds_to_plot,cmap="Blues",s=5,alpha=.4)
        plt.clim([-1,1])
        plt.title('predicted in dark blue')
        plt.savefig('../data/results/'+sub_file_name+'predicted_jumps.png')

        # get jump bout numbers then save out prediction csv
        jump_bouts = [0]
        bout_num=0
        for idx in np.arange(1,len(preds_to_plot)):
            if preds_to_plot[idx]==0:
                jump_bouts.append(0)
            elif preds_to_plot(idx-1)==0:
                bout_num+=1
                jump_bouts.append(bout_num)
            else:
                jump_bouts.append(bout_num)

        prediction = aligned_df.copy()
        prediction['predicted_jumps']=preds_to_plot
        prediction['jump_bouts']=jump_bouts
        prediction.to_csv('../data/results/'+sub_file_name+"_predictions.csv")
        print('===== finished all analysis for {}, starting next file'.format(sub_file_name))


    # make a dataframe with all summary data, save as csv
    column_vals=['sub_file_name','total_frames','num_entrances','first_frame_in_center','frames_with_nose','frames_with_head','frames_with_body','frames_within_50mm','fraction_frames_within_50mm','fraction_LIGHT_frames_within_50mm','fraction_LIGHT_frames_in_center','num_jumps']
    zipped=list(zip(full_path_list,total_frames,num_entrances,first_frame_in_center,frames_with_nose,frames_with_head,frames_with_body,frames_within_50mm, fraction_frames_within_50mm,fraction_LIGHT_frames_within_50mm,fraction_LIGHT_frames_in_center,num_jumps))
    summary_df=pd.DataFrame(zipped,columns=column_vals)
    summary_df.to_csv('../data/results/'+sub_file_name+"_summary.csv")








#
