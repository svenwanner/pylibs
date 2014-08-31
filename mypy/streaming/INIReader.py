import os
import numpy as np

from ConfigParser import ConfigParser
from mypy.tools.cg._transformations import euler_matrix

class Parameter(ConfigParser):

    def __init__(self, filepath=None):
        ConfigParser.__init__(self)

        self.inifile_loc = filepath

        # camera properties
        self.fov = None
        self.resolution_yx = None
        self.focal_length_mm = None
        self.focal_length_px = None
        self.sensor_width_mm = None
        self.sensor_heigth_mm = None
        self.euler_rotation_xyz = None
        self.final_camera_pos_m = None
        self.center_camera_pos_m = None
        self.initial_camera_pos_m = None
        self.camera_translation_vector = None
        self.image_files_location = None

        # scene properties
        self.baseline_m = None
        self.horopter_m = None
        self.max_baseline_mm = None
        self.number_of_sampling_points = None

        # computation properties
        self.inner_scale = None
        self.outer_scale = None
        self.min_coherence = None
        self.world_accuracy_m = None
        self.visible_world_area_m = None

        self.load(filepath)

    def __str__(self):
        str_out = "### camera ###\n"
        str_out += "fov : " + str(self.fov) + "\n"
        str_out += "focal_length_mm : " + str(self.focal_length_mm) + "\n"
        str_out += "focal_length_px : " + str(self.focal_length_px) + "\n"
        str_out += "sensor_width_mm : " + str(self.sensor_width_mm) + "\n"
        str_out += "sensor_heigth_mm : " + str(self.sensor_heigth_mm) + "\n"
        str_out += "resolution_yx : " + str(self.resolution_yx) + "\n"
        str_out += "initial_camera_pos_m : " + str(self.initial_camera_pos_m) + "\n"
        str_out += "euler_rotation_xyz : " + str(self.euler_rotation_xyz) + "\n"
        str_out += "camera_translation_vector : " + str(self.camera_translation_vector) + "\n"
        str_out += "final_camera_pos_m : " + str(self.final_camera_pos_m) + "\n"
        str_out += "center_camera_pos_m : " + str(self.center_camera_pos_m) + "\n"
        str_out += "image_files_location : " + self.image_files_location + "\n"
        str_out += "### scene ###\n"
        str_out += "number_of_sampling_points : " + str(self.number_of_sampling_points) + "\n"
        str_out += "baseline_m : " + str(self.baseline_m) + "\n"
        str_out += "max_baseline_mm : " + str(self.max_baseline_mm) + "\n"
        str_out += "horopter : " + str(self.horopter_m) + "\n"
        str_out += "### computation ###\n"
        str_out += "inner_scale : " + str(self.inner_scale) + "\n"
        str_out += "outer_scale : " + str(self.outer_scale) + "\n"
        str_out += "min_coherence : " + str(self.min_coherence) + "\n"
        str_out += "world_accuracy_m : " + str(self.world_accuracy_m) + "\n"
        str_out += "visible_world_area_m : " + str(self.visible_world_area_m) + "\n"

        return str_out



    def load(self, filepath):
        if filepath is not None:
            assert isinstance(filepath, str), "Invalid filepath type!"
            assert os.path.isfile(filepath), "Could not open ini file!"
            self.inifile_loc = filepath
            self.read(filepath)
            self.computeMissingParameter()

    def compute_look_at(self):
        """
        computes the cameras look at vector
        """
        look_at = np.mat([0.0, 0.0, -1.0, 0.0])
        rotation_matrix = euler_matrix(self.euler_rotation_xyz[0], self.euler_rotation_xyz[1], self.euler_rotation_xyz[2], axes='sxyz')
        look_at = look_at * np.linalg.inv(np.mat(rotation_matrix))
        look_at = np.array(look_at[0, 0:3])
        return look_at

    def computeMissingParameter(self):

        # set parameter obligatory
        self.image_files_location = self.get('camera', 'image_files_location')
        if not self.image_files_location.endswith(os.sep):
            self.image_files_location += os.sep
        if not os.path.exists(self.image_files_location):
            if not self.image_files_location.startswith(os.sep):
                self.image_files_location = os.sep + self.image_files_location
            self.image_files_location = os.path.dirname(self.inifile_loc) + self.image_files_location
            
        self.focal_length_mm = float(self.get('camera', 'focal_length_mm'))
        self.sensor_width_mm = float(self.get('camera', 'sensor_width_mm'))
        self.resolution_yx = np.array([int(self.get('camera', 'resolution_y')),
                                       int(self.get('camera', 'resolution_x'))])
        self.initial_camera_pos_m = np.array([float(self.get('camera', 'initial_camera_pos_x_m')),
                                              float(self.get('camera', 'initial_camera_pos_y_m')),
                                              float(self.get('camera', 'initial_camera_pos_z_m'))])
        self.euler_rotation_xyz = np.array([float(self.get('camera', 'euler_rotation_x_rad')),
                                            float(self.get('camera', 'euler_rotation_y_rad')),
                                            float(self.get('camera', 'euler_rotation_z_rad'))])
        self.camera_translation_vector_xyz = np.array([float(self.get('camera', 'camera_translation_vector_x_m')),
                                                      float(self.get('camera', 'camera_translation_vector_y_m')),
                                                      float(self.get('camera', 'camera_translation_vector_z_m'))])

        self.number_of_sampling_points = int(self.get('scene', 'number_of_sampling_points'))
        self.baseline_mm = float(self.get('scene', 'baseline_mm'))
        self.horopter_m = float(self.get('scene', 'horopter_m'))

        self.inner_scale = float(self.get('computation', 'inner_scale'))
        self.outer_scale = float(self.get('computation', 'outer_scale'))
        self.min_coherence = float(self.get('computation', 'min_coherence'))


        # compute the field of view of the camera
        self.fov = np.arctan2(self.sensor_width_mm/2.0, self.focal_length_mm)
        # compute focal length is pixel
        self.focal_length_px = self.focal_length_mm/self.sensor_width_mm * float(self.resolution_yx[1])
        # compute the maximum traveling distance of the camera
        self.max_baseline_m = float((self.number_of_sampling_points-1))*self.baseline_mm/1000.0
        # compute the final amd center camera position and the center position of the camera track
        self.final_camera_pos_m = self.initial_camera_pos_m + self.max_baseline_m * self.camera_translation_vector_xyz
        self.center_camera_pos_m = self.initial_camera_pos_m + self.max_baseline_m/2.0 * self.camera_translation_vector_xyz
        # compute field of view height
        self.sensor_heigth_mm = self.sensor_width_mm * float(self.resolution_yx[0])/float(self.resolution_yx[1])
        fov_h = np.arctan2(self.sensor_heigth_mm/2.0, self.focal_length_mm)
        # compute real visible scene width and height
        look_at = self.compute_look_at()
        distance = np.sqrt(np.sum((look_at[:]*self.horopter_m)**2))
        vsw = self.max_baseline_m + 2.0 * np.tan(self.fov) * distance
        vsh = 2.0 * np.tan(fov_h) * distance
        self.visible_world_area_m = [vsw, vsh]

        # if no world accuracy was set, compute optimal world accuracy
        opt_wa = 0
        if self.world_accuracy_m is None or self.world_accuracy_m <= 0.0:
            opt_wa = vsh/self.resolution_yx[0]
            self.world_accuracy_m = opt_wa
        # if world accuracy is set smaller than optimal set to optimal
        elif self.world_accuracy_m < opt_wa:
            self.world_accuracy_m = opt_wa




if __name__ == "__main__":
    p = Parameter()
    p.load("/home/swanner/Desktop/BusinessDemo/render/measurement/cam_000_000.ini")
    print p
