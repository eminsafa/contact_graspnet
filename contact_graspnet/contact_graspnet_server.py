import json
import os
import sys
import numpy as np

from contact_grasp_estimator import GraspEstimator
from data import load_available_input_data
from flask import Flask, request
import tensorflow.compat.v1 as tf


tf.disable_eager_execution()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR))

file = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'global_config.json'), 'r+')
global_config = json.loads(file.read())

checkpoint_dir = 'checkpoints/scene_test_2048_bs3_hor_sigma_001'
png_path = ''
K = None
z_range = [0.2, 1.1]
local_regions = False
filter_grasps = False
skip_border_objects = False
forward_passes = 5
segmap_id = 0
arg_configs = []

app = Flask(__name__)

# Build the model
grasp_estimator = GraspEstimator(global_config)
grasp_estimator.build_network()

# Add ops to save and restore all the variables.
saver = tf.train.Saver(save_relative_paths=True)

# Create a session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.Session(config=config)

# Load weights
grasp_estimator.load_weights(sess, saver, checkpoint_dir, mode='test')


@app.route('/run')
def read_file():
    path = request.args.get('path')
    if path:
        if os.path.exists(path) and os.path.isfile(path):
            print('Loading ', path)

            pc_segments = {}
            segmap, rgb, depth, cam_K, pc_full, pc_colors = load_available_input_data(path, K=K)

            if segmap is None and (local_regions or filter_grasps):
                raise ValueError('Need segmentation map to extract local regions or filter grasps')

            if pc_full is None:
                print('Converting depth to point cloud(s)...')
                pc_full, pc_segments, pc_colors = grasp_estimator.extract_point_clouds(
                    depth,
                    cam_K,
                    segmap=segmap,
                    rgb=rgb,
                    skip_border_objects=skip_border_objects,
                    z_range=z_range,
                )

            print('Generating Grasps...')
            pred_grasps_cam, scores, contact_pts, _ = grasp_estimator.predict_scene_grasps(
                sess,
                pc_full,
                pc_segments=pc_segments,
                local_regions=local_regions,
                filter_grasps=filter_grasps,
                forward_passes=forward_passes,
            )

            saved_path = 'results/predictions_{}'.format(
                os.path.basename(path.replace('png', 'npz').replace('npy', 'npz'))
            )
            np.savez(
                saved_path,
                pred_grasps_cam=pred_grasps_cam,
                scores=scores,
                contact_pts=contact_pts,
            )

            # Visualize results
            # show_image(rgb, segmap)
            # visualize_grasps(pc_full, pred_grasps_cam, scores, plot_opencv_cam=True, pc_colors=pc_colors)

            result = os.path.join(os.getcwd(), saved_path)
            print(f"Result: {result}")
            return result
        else:
            return f"File not found at path: {path}"
    else:
        return "Please provide a valid 'path' parameter."


if __name__ == '__main__':
    app.run()
