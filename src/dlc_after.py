import deeplabcut, os


video_folder = '/home/dennislab2/Desktop/videos/2022_SF_videos'
config = '/home/dennislab2/Desktop/DLC/2022_SF-thedennislab-2022-08-02/config.yaml'
list_of_files = [os.path.join(video_folder,file) for file in os.listdir(video_folder) if '.AVI' in file]

#deeplabcut.convert_detections2tracklets(config, list_of_files)

for video_file in list_of_files:
    print(video_file)
    success=0
    while success==0:
        for n in [10,9,8,7,6,5,4,3,2,1,0]:
            try:
                deeplabcut.stitch_tracklets(config,video_file,n_tracks=n)
                success=1
                print('changed success to 1')
            except:
                print('{} tracklets failed, trying next value'.format(n))
        if n==0:
            success=2




#At the very end
deeplabcut.analyze_videos_converth5_to_csv(video_folder,videotype='.AVI')
