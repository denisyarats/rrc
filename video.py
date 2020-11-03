import os
import sys

import imageio
import numpy as np

import utils
from trifinger_simulation import camera


class CustomTriFingerCameras:
    """Simulate the three cameras of the TriFinger platform."""
    def __init__(self, height, width):
        self.cameras = [
            # camera60
            camera.Camera(
                camera_position=[0.2496, 0.2458, 0.4190],
                camera_orientation=[0.3760, 0.8690, -0.2918, -0.1354],
                image_size=(width, height),
            )
        ]

    def get_images(self):
        """Get images.

        Returns:
            List of images, one per camera.  Order is [camera60, camera180,
            camera300].  See Camera.get_image() for details.
        """
        return [c.get_image() for c in self.cameras]


class VideoRecorder(object):
    def __init__(self, root_dir, height=256, width=256, fps=50):
        self.save_dir = utils.make_dir(root_dir, 'video') if root_dir else None
        self.camera = CustomTriFingerCameras(height, width)
        self.fps = fps
        self.frames = []
        self.count = 0

    def init(self, enabled=True):
        self.frames = []
        self.count = 0
        self.enabled = self.save_dir is not None and enabled

    def record(self):
        if self.enabled and self.count % self.fps == 0:
            images = self.camera.get_images()
            frame = np.concatenate(images, axis=1)
            self.frames.append(frame)
        self.count += 1

    def save(self, file_name):
        if self.enabled:
            path = os.path.join(self.save_dir, file_name)
            imageio.mimsave(path, self.frames)
