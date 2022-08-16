"""
Script to identify how many distinct calibrations we have.
Outputs a json file with:
[
    {
        R: (4, 3, 3),
        T: (4, 3, 1),
        f: (4, 1, 2),
        c: (4, 1, 2),
        trials: (1, <num_of_trials>)
    }
]
"""

import os
from glob import glob
import json
import logging
import numpy as np
import cv2
from scipy.spatial.transform import Rotation


PATH_TO_TRIALS = '/archive/2605_di_export'
KEY_2_COMPARE = ["R", "T", "f", "c"]

def rot_trans_to_homogenous(rot, trans):
    """
    Args
        rot: 3x3 rotation matrix
        trans: 3x1 translation vector
    Returns
        4x4 homogenous matrix
    """
    X = np.zeros((4, 4))
    X[:3, :3] = rot
    X[:3, 3] = trans.T
    X[3, 3] = 1
    return X


def homogenous_to_rot_trans(X):
    """
    Args
        x: 4x4 homogenous matrix
    Returns
        rotation, translation: 3x3 rotation matrix, 3x1 translation vector
    """

    return X[:3, :3], X[:3, 3].reshape(3, 1)


def rotation_to_homogenous(vec):
    rot_mat = Rotation.from_rotvec(vec)
    swap = np.identity(4)
    swap = np.zeros((4, 4))
    swap[:3, :3] = rot_mat.as_matrix()
    swap[3, 3] = 1
    return swap


def camera_params(path_to_workflow_phase):
    cam_parameters = {
        'R' : np.zeros((4, 3, 3)),
        'T' : np.zeros((4, 3, 1)),
        'f' : np.zeros((4, 1, 2)),
        'c' : np.zeros((4, 1, 2)),
        'trials' : []
    }
    camera_paths = list(sorted(glob(path_to_workflow_phase + '/*')))
    for i, camera_path in enumerate(camera_paths):

        # intrinsics

        intrinsics = os.path.join(camera_path, 'camera_calibration.yml')
        assert os.path.exists(intrinsics)
        fs = cv2.FileStorage(intrinsics, cv2.FILE_STORAGE_READ)
        logging.debug(f"Successfully opened {intrinsics}")

        color_intrinsics = fs.getNode("undistorted_color_camera_matrix").mat()
        cam_parameters['f'][i] = np.array([color_intrinsics[0, 0], color_intrinsics[1, 1]])
        cam_parameters['c'][i] = np.array([color_intrinsics[0, 2], color_intrinsics[1, 2]])

        depth2color_r = fs.getNode('depth2color_rotation').mat()
        depth2color_t = fs.getNode('depth2color_translation').mat() / 1000
        depth2color = rot_trans_to_homogenous(depth2color_r, depth2color_t.reshape(3))


        # extrinsics
        extrinsics = os.path.join(camera_path, "world2camera.json")
        assert os.path.exists(extrinsics)
        f = open(extrinsics, 'r')
        ext = json.load(f)
        logging.debug(f"Successfully opened {extrinsics}")
        trans = np.array([x for x in ext['translation'].values()])
        _R = ext['rotation']
        rot = Rotation.from_quat([_R['x'], _R['y'], _R['z'], _R['w']]).as_matrix()
        ext_homo = rot_trans_to_homogenous(rot, trans)

        yz_flip = rotation_to_homogenous(np.pi * np.array([1, 0, 0]))
        YZ_SWAP = rotation_to_homogenous(np.pi/2 * np.array([1, 0, 0]))

        depth2world = YZ_SWAP @ ext_homo @ yz_flip
        color2world = depth2world @ np.linalg.inv(depth2color)

        cam_parameters['R'][i], cam_parameters['T'][i] = homogenous_to_rot_trans(color2world)


    return cam_parameters


def compare_two_params(param_1, param_2):
    for key in KEY_2_COMPARE:
        if not np.array_equal(param_1[key], param_2[key]):
            return False
    
    return True


def traverse_trials():
    # List of dicts that consist of the cam params
    result = []
    for animal_trial in list(sorted(os.listdir(PATH_TO_TRIALS))):
        logging.debug(f"Parsing {animal_trial}")
        for workflow_phase in os.listdir(os.path.join(PATH_TO_TRIALS, animal_trial)):
            logging.debug(f"Parsing {workflow_phase}")
            path_to_workflow_phase = os.path.join(PATH_TO_TRIALS, animal_trial, workflow_phase)
            cam_parameters = camera_params(path_to_workflow_phase)

            added = False
            for existing_params in result:
                if compare_two_params(existing_params, cam_parameters):
                    existing_params["trials"].append(f"{animal_trial}_{workflow_phase}")
                    added = True
                    logging.debug(f"Added {animal_trial}_{workflow_phase} to an existing trial")

            if not added:
                cam_parameters["trials"].append(f"{animal_trial}_{workflow_phase}")
                result.append(cam_parameters)
                logging.debug(f"Added {animal_trial}_{workflow_phase} as new camera parameters")

        
        logging.info(f"Finished parsing {animal_trial}")

    
    with open('tmp_output/calibrations.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)

        logging.info(f"Saved calibrations in file tmp_output/calibrations.json")
    
    logging.info(f"There has been {len(result)} different calibrations")


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def main():
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    traverse_trials()


if __name__=='__main__':
    main()

