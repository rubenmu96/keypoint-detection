
class VideoPredictor:
    def __init__(self, model, cfg, video_processor=None, fps=30):
        self.model = model
        self.cfg = cfg
        self.video_processor = video_processor
        self.fps = fps

        self.model = self.model.eval()


    def get_youtube_video(self, url):
        # to some magic with video_processor
        ...


    def predict_frame(self):
        ...


    def predict(self, url):
        ...