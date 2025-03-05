import os
import cv2
import numpy as np
from tqdm import tqdm

# Set environment variables for OpenCV
os.environ['OPENCV_LOG_LEVEL'] = 'FATAL'
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = "-8"

class ImageStack:
    def __init__(self, filename: str, channel=None):
        self.filename = filename

        # load the video file in a cv2 object
        self.video = cv2.VideoCapture(filename)

        if not self.video.isOpened():
            raise ValueError('File path likely incorrect, failed to open.')

        property_id = int(cv2.CAP_PROP_FRAME_COUNT)

        # get the number of frames
        self.frame_count = int(self.video.get(property_id))
        # get the fps
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        # store the specified colour channel (if any)
        self.channel = channel

        # read first frame to determine the shape (keep original shape)
        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.shape = self[0].shape

    def __len__(self):
        return self.frame_count
            
    def __getitem__(self, t):
        """Fetches frame at specified index (can handle non-integer index)"""
        # handle negative indices
        if t < 0:
            t= len(self) + t - 1
        
        # check index is in range
        assert t < self.frame_count
        self.video.set(cv2.CAP_PROP_POS_FRAMES, t - 1)
        success, image = self.video.read()

        if self.channel is not None:
            return image[...,self.channel]
        if image is not None:
            return image.mean(axis=2).astype(int)
        self.shape = self[0].shape
    
    def pre_load_stack(self, renormalise=False):
        """Load all frames into a numpy array which is pickleable."""
        # load the first frame to determine whether it is RGB or grayscale
        first_frame = self[0]

        # handle grayscale or RGB frames (2 or 3 dimensions)
        if len(first_frame.shape) == 2:  # Grayscale
            frames = np.zeros((self.frame_count, *first_frame.shape), dtype=np.float32)
        elif len(first_frame.shape) == 3:  # RGB
            frames = np.zeros((self.frame_count, *first_frame.shape), dtype=np.float32)
        else:
            raise ValueError(f"Unsupported frame shape: {first_frame.shape}")

        # load all frames into the pre-constructed array
        for i in tqdm(range(self.frame_count), desc="Pre-loading frames", unit="frame"):
            if renormalise:
                frames[i] = self[i] / (np.mean(self[i]))
            else:
                frames[i] = self[i]
            
        return frames
    
    def verify_frames(self):
        """Verifies that the frames stored in preloaded_stack match those from the stack object."""

        # load preloaded stack
        preloaded_stack = self.pre_load_stack()

        # loop through each frame and verify data
        for i in range(self.frame_count):
            # compare corresponding frames in the preloaded stack and the stack
            if not np.array_equal(preloaded_stack[i], self[i]):
                print(f"Frames at index {i} do not match!")
                return False
        
        print("All frames match correctly.")
        return True
