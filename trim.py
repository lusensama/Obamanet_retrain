from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip
from glob import glob

videos = glob('videos/*.mp4')
name = 1
for video_name in videos:
    t1 = 9
    video_clip = VideoFileClip(video_name)
    print( video_clip.duration )
    t2 = video_clip.duration - 18
    filename = 'videos_cut/{:05}.mp4'.format(name)
    print (filename)
    ffmpeg_extract_subclip(video_name, t1, t2, targetname=filename)
    name+=1