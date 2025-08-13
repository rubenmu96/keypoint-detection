import cv2
from pytubefix import YouTube
import validators

class VideoProcessor:
    """Youtube and mp4 video processor"""
    def __init__(self, input_path, video_name, output_path=None, folder=None, max_duration=None, start_time=None, end_time=None):
        # TODO: improve names
        self.input_path = input_path
        self.video_name = video_name
        self.output_path = output_path if folder is None else f"{folder}{output_path}"
        self.folder = folder # make sure folder ends with "/"?
        self.max_duration = max_duration
        self.start_time = start_time
        self.end_time = end_time

        if start_time is not None and end_time is None:
            self.end_time = float('inf')
        elif start_time is None and end_time is not None:
            self.start_time = 0

    @staticmethod
    def _contains_mp4(string):
        if string[-4:] == ".mp4":
            return string
        else:
            return string+".mp4"
        
    @staticmethod
    def _remove_mp4(string):
        if string[-4:] == ".mp4":
            return string[:-4]
        else:
            return string

    def get_yt_video(self, url, folder="examples/", name_of_video=None):
        youtube = YouTube(url)
        
        if name_of_video is not None:
            youtube.title = self._remove_mp4(name_of_video)
        video = youtube.streams.get_highest_resolution()
        video.download(folder)
        return self._contains_mp4(f"{folder}{youtube.title}")

    def extract_video_segment(self, input_path, output_path, start_time, end_time):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        start_frame = int(start_time * fps)
        end_frame = min(int(end_time * fps), total_frames)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self._contains_mp4(output_path), fourcc, fps, 
                            (int(cap.get(3)), int(cap.get(4))))
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for _ in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        
        cap.release()
        out.release()

    def shorten_video(self, input_path, output_path, max_duration):
        """Shorten video to specified duration (in seconds) from the start"""
        self.extract_video_segment(input_path, output_path, 0, max_duration)

    def __call__(self):
        if validators.url(self.input_path):
            input_path = self.get_yt_video(self.input_path, self.folder, self.video_name)
        else:
            input_path = self.input_path
        
        if self.start_time is not None or self.end_time is not None:
            self.extract_video_segment(input_path, self.output_path, self.start_time, self.end_time)
        elif self.max_duration is not None:
            self.shorten_video(input_path, self.output_path, self.max_duration)