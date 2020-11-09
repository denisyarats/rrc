import pybullet
from scipy.spatial.transform import Rotation as R
import numpy as np


class OrientationMarker:
    """Visualize a cube."""
    def __init__(
        self,
        length,
        radius,
        position,
        orientation,
        **kwargs,
    ):
        """
        Create a cube marker for visualization

        Args:
            width (float): Length of one side of the cube.
            position: Position (x, y, z)
            orientation: Orientation as quaternion (x, y, z, w)
            color: Color of the cube as a tuple (r, b, g, q)
        """

        self._kwargs = kwargs
        self._length = length
        colors = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]
        positions = self.compute_positions(position, orientation, length)

        self.body_ids = []
        for i in range(positions.shape[0]):
            shape_id = pybullet.createVisualShape(
                shapeType=pybullet.GEOM_SPHERE,
                length=length,
                radius=radius,
                rgbaColor=colors[i])
            body_id = pybullet.createMultiBody(baseVisualShapeIndex=shape_id,
                                               basePosition=positions[i],
                                               baseOrientation=[0, 0, 0, 1])
            self.body_ids.append(body_id)

    def compute_positions(self, position, orientation, magnitude):
        shifts = R.from_quat(orientation).apply(np.eye(3))
        shifts *= magnitude
        shifts += np.array(position)
        return shifts

    def set_state(self, position, orientation):
        positions = self.compute_positions(position, orientation, self._length)
        for i, body_id in enumerate(self.body_ids):
            pybullet.resetBasePositionAndOrientation(
                body_id,
                positions[i],
                [0, 0, 0, 1],
                **self._kwargs,
            )


class CubeMarker:
    """Visualize a cube."""
    def __init__(
        self,
        width,
        position,
        orientation,
        color=(1, 1, 0, 0.5),
        **kwargs,
    ):
        self._kwargs = kwargs

        self.shape_id = pybullet.createVisualShape(
            shapeType=pybullet.GEOM_BOX,
            halfExtents=[width / 2] * 3,
            rgbaColor=color,
            **self._kwargs,
        )
        self.body_id = pybullet.createMultiBody(
            baseVisualShapeIndex=self.shape_id,
            basePosition=position,
            baseOrientation=orientation,
            **self._kwargs,
        )

    def set_state(self, position, orientation):
        pybullet.resetBasePositionAndOrientation(
            self.body_id,
            position,
            orientation,
            **self._kwargs,
        )
