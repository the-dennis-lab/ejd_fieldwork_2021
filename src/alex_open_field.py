"""alex_open_field.py:
REQUIRED INPUTS:
    1. a folder or file

OPTIONAL INPUTS:
    2. a trained .pkl file or tries to find one in ../data.
    3. a csv that has 'trim times' for videos that do not start at the 'correct' moment (camera started early)
    4. a step number (int)
        #0 for all steps,
        #1 loads DLC file & reformats: smooths and removes low confidence points
        #2 takes existing smoothed file, loads, and aligns to box with coordinates 0,0 0,500 500,0 500,500
        #3 for taking existing adjusted points file, loading, and
        #4 for running the prediction on aligned, adjusted files
    5. rescale (0 or 1)- if rescaling some columns, set to 1, default is 0
HARDCODED:
    a few things are currently hardcoded for convenience.
    - If using a file to subset the dataframes by start/end time,
    then we assume you're using filenames like VID_20210925_134723 for your videos, that have 18 long strings. Else,
    search for 'os.path.basename(filename)[0:19]'mil and change 18 to a number of your choosing.
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
import glob, os, csv, sys, cv2, math, itertools, joblib, datetime
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
        box_likelihood = 0.95
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
                if np.max(part.likelihood) < box_likelihood:
                    box_likelihood=np.max(part.likelihood)-0.02
                part.x[part.likelihood < box_likelihood]=np.nan
                part.y[part.likelihood < box_likelihood]=np.nan
                part.x = part.x.interpolate()
                part.x=part.x.rolling(seconds*fps,min_periods=1).median()
                part.y = part.y.interpolate()
                part.y=part.y.rolling(seconds*fps,min_periods=1).median()
                new_location = pd.concat({col:part},axis=1)
                new_df=pd.concat([new_df,new_location],axis=1)

    return new_df


def align_df(df):
    cols=df.columns
    list_of_cols = [col[0] for col in cols if "Unnamed" in col[0]]
    if len(list_of_cols)>0:
        df=df.drop(columns=list_of_cols)
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
    l_frames=[]
    frames_within_50mm=[]
    fraction_frames_within_50mm=[]
    fraction_LIGHT_frames_within_50mm=[]
    fraction_LIGHT_frames_in_center=[]
    first_frame_in_center=[]
    num_entrances=[]
    total_frames=[]
    num_jumps=[]
    num_rears=[]
    num_frames_jumping=[]
    num_frames_rearing=[]
    first_nose_in_box=[]
    first_head_in_box=[]
    first_body_in_box=[]
    sum_dist=[]
    sum_dist_without_jumps=[]

    if not os.path.isdir('../data/results'):
        os.mkdir('../data/results')

    # arg parser
    file_or_folder=sys.argv[1]
    if os.path.isdir(file_or_folder):
        folder_of_files = file_or_folder
        folder_list = os.listdir(file_or_folder)
        full_path_list = [os.path.join(file_or_folder,file) for file in folder_list if 'csv' in file]
    elif os.path.isfile(file_or_folder):
        full_path_list = [file_or_folder]
    else:
        print('after alex_open_field.py, please provide a file or folder. You typed {} and this does not exist'.format(file_or_folder))

    if len(sys.argv) > 2 and os.path.isfile(sys.argv[2]):
        trained_file = sys.argv[2]
    elif os.path.isfile('../data/20230305_RF_jumps_rears.pkl'):
        trained_file = '../data/20230305_RF_jumps_rears.pkl'
    else:
        print('there is no trained .pkl file in ../data and you did not provide a path to one. Please provide a path to one')

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

    if len(sys.argv) > 4:
        step = int(sys.argv[4])
    else:
        step=0 # run all

    rescale=0
    if len(sys.argv) > 5 and sys.argv[5]==1:
        rescale = 1
    else:
        print('not rescaling')
    # load pickle trained file
    clf = joblib.load(trained_file)

    print('===== loaded everything successfully, now analyzing video csv(s)')


    # get a date and time for saving summary data
    today = datetime.datetime.today()

    for filename in full_path_list:
        print('on {}'.format(filename))
        sub_file_name=os.path.basename(filename)[0:19]

        if step == 0 or step==1:
            print('on step 1/4')
            # load df from DLC, reformat with smoothing
            df = pd.read_csv(filename,header=[1,2])
            new_df=reformat_df(df,0.9,1000,15,3,30)

            if to_subset:
                try:
                    times_vals=times[times.File_name==sub_file_name]
                    start_val=round(list(times_vals.Start_time_in_seconds)[0]*FPS)
                    end_val=round(list(times_vals.End_time_in_seconds)[0]*FPS)
                    new_df=new_df[new_df.index<end_val]
                    new_df=new_df[new_df.index>start_val]
                    new_df=new_df.reset_index()
                    new_df=new_df.drop(columns=['index'])
                except:
                    print('failed to subset filename, skipping subsetting')
            new_df.to_csv(os.path.join('..','data','results',sub_file_name+"_adj.csv"))

            plt.scatter(new_df.box_bl['x'],new_df.box_bl['y'],c='r') #0,500mm
            plt.scatter(new_df.box_br['x'],new_df.box_br['y'],c='pink') #500,500mm
            plt.scatter(new_df.box_tl['x'],new_df.box_tl['y'],c='b') #0,0mm
            plt.scatter(new_df.box_tr['x'],new_df.box_tr['y'],c='cyan') #500,0mm
            plt.savefig(os.path.join('..','data','results',sub_file_name+"_adj_points.png"))
            plt.close()

            print('===== saved out {} adjusted csv, plotting then aligning next'.format(sub_file_name))

        if step==0 or step==2:
            try:
                new_df=pd.read_csv(filename[0:-4]+"_adj.csv",header=[0,1])
                new_df = pd.read_csv(os.path.join('..','data','results',sub_file_name+"_adj.csv"),header=[0,1])
            except:
                print('aligned dataframe not found at {}'.format(os.path.join('..','data','results',sub_file_name+"_adj.csv")))
            print('starting step 2/4')
            # affine transform to correct for sliding FOV
            aligned_df=align_df(new_df)
            aligned_df.to_csv(os.path.join('..','data','results',sub_file_name+"_aligned.csv"))
            print('===== saved out {} aligned csv, plotting then getting summary data'.format(sub_file_name))

            # plot newly aligned points
            plt.scatter(aligned_df.nose['x'],aligned_df.nose['y'],c='k',s=1,alpha=0.5)
            plt.scatter(aligned_df.box_bl['x'],aligned_df.box_bl['y'],c='r')
            plt.scatter(aligned_df.box_br['x'],aligned_df.box_br['y'],c='pink')
            plt.scatter(aligned_df.box_tl['x'],aligned_df.box_tl['y'],c='b')
            plt.scatter(aligned_df.box_tr['x'],aligned_df.box_tr['y'],c='cyan')
            plt.savefig(os.path.join('..','data','results',sub_file_name+"_all_points_with_corners.png"))
            plt.close()

        if step==0 or step==3:
            try:
                aligned_df = pd.read_csv(os.path.join('..','data','results',sub_file_name+"_aligned.csv"),header=[0,1])
            except:
                print('cannot open aligned_df at {}'.format(os.path.join('..','data','results',sub_file_name+"_aligned.csv")))
            # get all the identified nose/ear/tailbase points (if not foudn with
            #sufficient confidence, DLC gives a NAN, this ignores the NANs by making
            # them zeros, and all non-nans as ones)
            nose_vals = ~np.isnan(aligned_df.nose['x'])
            ear_l_vals = ~np.isnan(aligned_df.ear_left['x'])
            ear_r_vals = ~np.isnan(aligned_df.ear_right['x'])
            tail_vals = ~np.isnan(aligned_df.tail_base['x'])

            # make a subset of just the location (x,y) and prediction value
            #(likelihood) of the animal's nose
            nose_df=aligned_df.nose
            nose_df['within50mm']=0
            for idx in nose_df.index:
                xval=nose_df.x[idx]
                yval=nose_df.y[idx]
                if xval < 50 or xval > 450 or yval < 50 or yval > 450:
                    nose_df.iloc[idx,3]=1 #3 is the column we're filling with ones
                    #if and only if the points are outside the box

            # get some summary data: # of frames with the nose in?
            # nose + ears (aka head)? nose + ears + tailbase? (aka body)
            # I chose these points because the paws and tail tip aren't always
            # picked up as well by DLC, so these are high-fidelity pts
            frames_with_nose.append(np.sum(nose_vals))
            # if the nose entered the box, find the first entrance frame, else: nan
            if frames_with_nose[-1]>0:
                first_nose_in_box.append(np.argmax(nose_vals))
            else:
                first_nose_in_box.append(np.argmax(nose_vals))
            frames_with_head.append(np.sum(nose_vals*ear_l_vals*ear_r_vals))
            # if the HEAD entered the box, find the first entrance frame, else: nan
            if frames_with_head[-1]>0:
                first_head_in_box.append(np.argmax(nose_vals*ear_l_vals*ear_r_vals))
            else:
                first_head_in_box.append(np.nan)
            # if the BODY entered the box, find the first entrance frame, else: nan
            frames_with_body.append(np.sum(nose_vals*ear_l_vals*ear_r_vals*tail_vals))
            if frames_with_body[-1]>0:
                first_body_in_box.append(np.argmax(nose_vals*ear_l_vals*ear_r_vals*tail_vals))
            else:
                first_body_in_box.append(np.nan)
            total_frames.append(len(nose_vals))

            # let's generate some more summary data:
            nose_sum=np.sum(nose_vals)
            summed_val=np.sum(nose_df.within50mm)
            # how many frames are within 50mm of the edges of the box? (inluding
            # those outside the edges which are jumps or rears)
            frames_within_50mm.append(summed_val)
            fraction_frames_within_50mm.append(summed_val/len(nose_df))
            fraction_LIGHT_frames_in_center.append((nose_sum-summed_val)/nose_sum)
            fraction_LIGHT_frames_within_50mm.append(summed_val/nose_sum)
            sub_nose_df=nose_df[~np.isnan(nose_df.x)]
            sub_nose_df=sub_nose_df[sub_nose_df.within50mm<1]
            try:
                first_frame_in_center.append(sub_nose_df.index[0])
            except:
                first_frame_in_center.append(np.nan)
            # add 2 cols to df: one for if the animal was in box or not, and one for
            #the "bout number" which gives us # of entrances
            # also keep track of inter-bout-intervals
            aligned_df['in_box']=float(0)
            aligned_df['light_bout_num']=float(0)
            inter_light_interval=np.zeros(len(aligned_df))
            inter_light_bout=0
            bout_num=0
            for idx in aligned_df.index:
                if np.isnan(aligned_df.nose.x[idx]):
                    aligned_df.iloc[idx,-2]=0
                    aligned_df.iloc[idx,-1]=0
                    if aligned_df.iloc[idx-1,-2]>0:
                        inter_light_bout+=1
                    inter_light_interval[idx]=inter_light_bout
                elif aligned_df.iloc[idx-1,-2]==0:
                    bout_num+=1
                    aligned_df.iloc[idx,-2]=1
                    aligned_df.iloc[idx,-1]=bout_num
                else:
                    aligned_df.iloc[idx,-2]=1
                    aligned_df.iloc[idx,-1]=bout_num
            num_entrances.append(np.max(aligned_df.light_bout_num))

            #plot inter light bout histogram
            plt.hist(inter_light_interval)#
            plt.savefig(os.path.join('..','data','results',sub_file_name+"_inter_light_bout_hist.png"))
            plt.close()

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
            plt.savefig(os.path.join('..','data','results',sub_file_name+"_all-points_color_within50mm.png"))
            plt.close()


            # add distances between animal's body points
            aligned_df['dists','nose_midpoint_entrance']=float(0)
            aligned_df['dists','nose_ear_right']=float(0)
            aligned_df['dists','nose_ear_left']=float(0)
            aligned_df['dists','nose_paw_left_back']=float(0)
            aligned_df['dists','nose_paw_left_front']=float(0)
            aligned_df['dists','nose_paw_right_back']=float(0)
            aligned_df['dists','nose_paw_right_front']=float(0)
            aligned_df['dists','nose_tail_tip']=float(0)
            aligned_df['dists','nose_tail_base']=float(0)
            aligned_df['speed']=float(0)
            nanmean_xs=np.zeros(len(aligned_df))
            nanmean_ys=np.zeros(len(aligned_df))
            animal_dist_traveled=np.zeros(len(aligned_df))

            entrance_x=np.nanmean([np.nanmedian(aligned_df.opening_top.x),np.nanmedian(aligned_df.opening_bottom.x[idx])])
            entrance_y=np.nanmean([np.nanmedian(aligned_df.opening_top.y),np.nanmedian(aligned_df.opening_bottom.y[idx])])

            ind_idx=-1
            for idx in aligned_df.index:
                ind_idx+=1
                rel_idx = idx-aligned_df.index[0]
                nanmean_xs[ind_idx]=np.nanmean(
                        [aligned_df.nose.x[idx],
                        aligned_df.ear_left.x[idx],
                        aligned_df.ear_right.x[idx],
                        aligned_df.tail_base.x[idx]])
                nanmean_ys[ind_idx]=np.nanmean(
                        [aligned_df.nose.y[idx],
                        aligned_df.ear_left.y[idx],
                        aligned_df.ear_right.y[idx],
                        aligned_df.tail_base.y[idx]])
                if rel_idx > 0:
                    speed = np.abs(math.dist([nanmean_xs[ind_idx],nanmean_ys[ind_idx]],[nanmean_xs[ind_idx-1],nanmean_ys[ind_idx-1]]))
                else:
                    speed=0
                animal_dist_traveled[ind_idx]=speed
                aligned_df.iloc[rel_idx,-1]=speed
                aligned_df.iloc[rel_idx,-2]=math.dist([aligned_df.nose.x[idx],aligned_df.nose.y[idx]],[aligned_df.tail_base.x[idx],aligned_df.tail_base.y[idx]])
                aligned_df.iloc[rel_idx,-3]=math.dist([aligned_df.nose.x[idx],aligned_df.nose.y[idx]],[aligned_df.tail_tip.x[idx],aligned_df.tail_tip.y[idx]])
                aligned_df.iloc[rel_idx,-4]=math.dist([aligned_df.nose.x[idx],aligned_df.nose.y[idx]],[aligned_df.paw_right_front.x[idx],aligned_df.paw_right_front.y[idx]])
                aligned_df.iloc[rel_idx,-5]=math.dist([aligned_df.nose.x[idx],aligned_df.nose.y[idx]],[aligned_df.paw_right_back.x[idx],aligned_df.paw_right_back.y[idx]])
                aligned_df.iloc[rel_idx,-6]=math.dist([aligned_df.nose.x[idx],aligned_df.nose.y[idx]],[aligned_df.paw_left_front.x[idx],aligned_df.paw_left_front.y[idx]])
                aligned_df.iloc[rel_idx,-7]=math.dist([aligned_df.nose.x[idx],aligned_df.nose.y[idx]],[aligned_df.paw_left_back.x[idx],aligned_df.paw_left_back.y[idx]])
                aligned_df.iloc[rel_idx,-8]=math.dist([aligned_df.nose.x[idx],aligned_df.nose.y[idx]],[aligned_df.ear_left.x[idx],aligned_df.ear_left.y[idx]])
                aligned_df.iloc[rel_idx,-9]=math.dist([aligned_df.nose.x[idx],aligned_df.nose.y[idx]],[aligned_df.ear_right.x[idx],aligned_df.ear_right.y[idx]])
                aligned_df.iloc[rel_idx,-10]=math.dist([aligned_df.nose.x[idx],aligned_df.nose.y[idx]],[entrance_x,entrance_y])
            aligned_df['inter_light_bout']=inter_light_interval
            aligned_df['nanmean_x']=nanmean_xs
            aligned_df['nanmean_y']=nanmean_ys
            aligned_df['animal_dist_traveled']=animal_dist_traveled
            aligned_df.to_csv(os.path.join('..','data','results',sub_file_name+"_aligned_dists.csv"))
            print("===== aligned df saved out, on to predicting jumps")

        if step==0 or step==4:
            try:
                aligned_df = pd.read_csv(os.path.join('..','data','results',sub_file_name+"_aligned_dists.csv"),header=[0,1])
            except:
                print('cannot find aligned_df file for step 4 at {}'.format(os.path.join('..','data','results',sub_file_name+"_aligned_dists.csv")))
            # select these columns for predicting jumps (used for training)
            df_for_prediction=pd.concat([aligned_df['nose']['x'],aligned_df['nose','y'],aligned_df['nose']['likelihood'],aligned_df['tail_base']['likelihood'],aligned_df['paw_left_back']['likelihood'],aligned_df['paw_right_back']['likelihood'],aligned_df['dists']['nose_midpoint_entrance'],aligned_df['dists']['nose_tail_base'],aligned_df['speed']],axis=1)
            df_for_prediction.columns=['nose_x','nose_y','nose_likelihood','tail_base_likelihood','paw_lb_likelihood','paw_rb_likelihood','dists_fromentrance','dists_nose_tail_base','speed']
            try:
                cols=aligned_df.animal_dist_traveled.columns[0]
                animal_dist_traveled=aligned_df.animal_dist_traveled[cols]
            except:
                animal_dist_traveled=aligned_df['animal_dist_traveled']


            # if rescaling, rescale only the large value columns, so all columns are between 0, 1
            if rescale:
                for col in ['nose_x','nose_y','dists_fromentrance','dists_nose_tail_base','speed']:
                    min_val=np.nanmin(df_for_prediction[col])
                    if min_val < 0:
                        add_val=np.abs(min_val)
                    else:
                        add_val=0
                    maxval=np.nanmax(df_for_prediction[col])
                    df_for_prediction[col]=np.add(df_for_prediction[col],add_val)/np.nanmax(df_for_prediction[col])

            preds_to_plot=clf.predict(df_for_prediction.fillna(0))

            plt.scatter(aligned_df.nose.x,aligned_df.nose.y,c=preds_to_plot,cmap="Blues",s=5,alpha=.4)
            plt.clim([-1,1])
            plt.title('predicted in dark blue')
            plt.savefig(os.path.join('..','data','results',sub_file_name+'_predicted_jumps.png'))
            plt.close()

            # get jump bout numbers then save out prediction csv
            # but combine jumps if they are closer than 2 frames
            jump_bouts = np.zeros(len(preds_to_plot))
            jump_bout_num=0
            rear_bouts = np.zeros(len(preds_to_plot))
            rear_bout_num=0

            for idx in np.arange(4,len(preds_to_plot)):
                if preds_to_plot[idx]==1:
                    if 1 not in np.unique(preds_to_plot[idx-4:idx]):
                        jump_bout_num+=1
                    jump_bouts[idx]=jump_bout_num
                elif preds_to_plot[idx]==2:
                    if 2 not in np.unique(preds_to_plot[idx-4:idx]):
                        rear_bout_num+=1
                    rear_bouts[idx]=rear_bout_num

            # now remove jumps that don't have at least 4 frames
            # and get inter-jump-bouts
            vals_to_rm=[]
            rear_vals_to_rm=[]
            animal_dist_traveled_without_jumps=np.abs(animal_dist_traveled.copy())
            jump_bouts_list=list(jump_bouts)
            rear_bouts_list=list(rear_bouts)
            inter_jump_bout_num=0
            inter_jump_bouts=np.zeros(len(preds_to_plot))
            for val in np.unique(rear_bouts):
                if rear_bouts_list.count(val) < 4:
                    rear_vals_to_rm.append(val)
            for idx in np.arange(0,len(rear_bouts)):
                if rear_bouts[idx] in rear_vals_to_rm:
                    rear_bouts[idx]=0

            for val in np.unique(jump_bouts):
                if jump_bouts_list.count(val) < 4:
                    vals_to_rm.append(val)
            for idx in np.arange(0,len(jump_bouts)):
                if jump_bouts[idx] in vals_to_rm:
                    jump_bouts[idx]=0
                    animal_dist_traveled_without_jumps[idx]=0
                if idx < len(jump_bouts)-1 and jump_bouts[idx]>0:
                    if jump_bouts[idx+1]==0:
                        inter_jump_bout_num+=1
                    else:
                        inter_jump_bouts[idx]=inter_jump_bout_num

            # get the number of unique 4+ length jumps
            num_jumps.append(len(np.unique(jump_bouts)))
            num_rears.append(len(np.unique(rear_bouts)))
            num_frames_jumping.append(len(jump_bouts[jump_bouts>0]))
            num_frames_rearing.append(len(rear_bouts[rear_bouts>0]))
            prediction = aligned_df.copy()
            prediction['predicted_jumps']=preds_to_plot
            prediction['jump_bouts']=jump_bouts
            prediction['rear_bouts']=rear_bouts

            # get distance traveled under different jump inclusion/exclusions
            prediction['inter_jump_bouts']=inter_jump_bouts

            #save out prediction csv
            prediction['animal_dist_without_jumps']=animal_dist_traveled_without_jumps
            sum_dist_without_jumps.append(np.sum(animal_dist_traveled_without_jumps))
            sum_dist.append(np.sum(np.abs(animal_dist_traveled)))
            prediction.to_csv(os.path.join('..','data','results',sub_file_name+"_predictions.csv"))
            print('===== finished all analysis for {}, saving intermediate summary data then starting next file'.format(sub_file_name))

            # make a dataframe with all summary data, save as csv
            column_vals=['sub_file_name','total_frames','num_entrances','first_frame_with_nose','first_frame_with_head','first_frame_with_body','first_frame_in_center','frames_with_nose','frames_with_head','frames_with_body','frames_within_50mm','fraction_frames_within_50mm','fraction_LIGHT_frames_within_50mm','fraction_LIGHT_frames_in_center','num_jumps','sum_dist','sum_dist_without_jumps','num_frames_jumping','num_rears','num_frames_rearing']
            zipped=zip(full_path_list,total_frames,num_entrances,first_nose_in_box,first_head_in_box,first_body_in_box,first_frame_in_center,frames_with_nose,frames_with_head,frames_with_body,frames_within_50mm, fraction_frames_within_50mm,fraction_LIGHT_frames_within_50mm,fraction_LIGHT_frames_in_center,num_jumps,sum_dist,sum_dist_without_jumps,num_frames_jumping,num_rears,num_frames_rearing)
            summary_df=pd.DataFrame(zipped,columns=column_vals)
            summary_df.to_csv(os.path.join('..','data','results',today.strftime("%Y-%m-%d-%H%M%S")+"_summary.csv"))
if step==0 or step==4:
    # save final summary with date and time
    summary_df.to_csv(os.path.join('..','data','results',today.strftime("%Y-%m-%d-%H%M%S")+"_summary.csv"))



#
