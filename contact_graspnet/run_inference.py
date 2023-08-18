from inference import inference

global_config = {
    'DATA': {'gripper_width': 0.08, 'input_normals': False, 'use_uniform_quaternions': False, 'train_on_scenes': True,
             'labels': {'to_gpu': False,
                        'bin_weights': [0.16652107, 0.21488856, 0.37031708, 0.55618503, 0.75124664, 0.93943357,
                                        1.07824539, 1.19423112, 1.55731375, 3.17161779], 'contact_gather': 'knn',
                        'filter_z': True, 'k': 1, 'max_radius': 0.005, 'min_unique_pos_contacts': 1,
                        'num_neg_contacts': 0, 'num_pos_contacts': 10000,
                        'offset_bins': [0, 0.00794435329, 0.0158887021, 0.0238330509, 0.0317773996, 0.0397217484,
                                        0.0476660972, 0.055610446, 0.0635547948, 0.0714991435, 0.08], 'z_val': -0.1},
             'raw_num_points': 20000, 'ndataset_points': 20000, 'num_point': 2048, 'sigma': 0.001, 'clip': 0.005,
             'use_farthest_point': False, 'train_and_test': False, 'num_test_scenes': 1000, 'intrinsics': 'realsense',
             'classes': None},
    'LOSS': {'min_geom_loss_divisor': 1.0, 'max_geom_loss_divisor': 100.0, 'offset_loss_type': 'sigmoid_cross_entropy',
             'too_small_offset_pred_bin_factor': 0, 'topk_confidence': 512},
    'MODEL': {'bin_offsets': True, 'contact_distance_offset': True, 'dir_vec_length_offset': False,
              'grasp_conf_head': {'conv1d': 1, 'dropout_keep': 0.5},
              'grasp_dir_head': {'conv1d': 3, 'dropout_keep': 0.7}, 'joint_head': {'conv1d': 4, 'dropout_keep': 0.7},
              'joint_heads': False, 'larger_model': False, 'asymmetric_model': True, 'model': 'contact_graspnet',
              'pointnet_fp_modules': [{'mlp': [256, 256]}, {'mlp': [256, 128]}, {'mlp': [128, 128, 128]}],
              'pointnet_sa_module': {'group_all': True, 'mlp': [256, 512, 1024]}, 'pointnet_sa_modules_msg': [
            {'mlp_list': [[32, 32, 64], [64, 64, 128], [64, 96, 128]], 'npoint': 2048, 'nsample_list': [32, 64, 128],
             'radius_list': [0.02, 0.04, 0.08]},
            {'mlp_list': [[64, 64, 128], [128, 128, 256], [128, 128, 256]], 'npoint': 512,
             'nsample_list': [64, 64, 128], 'radius_list': [0.04, 0.08, 0.16]},
            {'mlp_list': [[64, 64, 128], [128, 128, 256], [128, 128, 256]], 'npoint': 128,
             'nsample_list': [64, 64, 128], 'radius_list': [0.08, 0.16, 0.32]}], 'pred_contact_approach': True,
              'pred_contact_base': True, 'pred_contact_offset': True, 'pred_contact_success': True,
              'pred_grasps_adds': False, 'pred_grasps_adds_gt2pred': False},
    'OPTIMIZER': {'adds_gt2pred_loss_weight': 1, 'adds_loss_weight': 10, 'approach_cosine_loss_weight': 1,
                  'batch_size': 5, 'bn_decay_clip': 0.99, 'bn_decay_decay_rate': 0.5, 'bn_decay_decay_step': 200000,
                  'bn_init_decay': 0.5, 'decay_rate': 0.7, 'decay_step': 200000, 'dir_cosine_loss_weight': 1,
                  'learning_rate': 0.001, 'max_epoch': 16, 'momentum': 0.9, 'offset_loss_weight': 1,
                  'optimizer': 'adam', 'score_ce_loss_weight': 1},
    'TEST': {'center_to_tip': 0.0, 'allow_zero_margin': 0, 'bin_vals': 'max', 'extra_opening': 0.005,
             'first_thres': 0.23, 'second_thres': 0.18, 'max_farthest_points': 150, 'num_samples': 200, 'save': False,
             'scale_fac': [1.25, 1.0, 0.75, 0.5], 'scales': False, 'with_replacement': False, 'filter_thres': 0.0001}}

ckpt_dir = 'checkpoints/scene_test_2048_bs3_hor_sigma_001'
np_path = '/home/juanhernandezvega/dev/contact_graspnet/test_data/0.npy'
png_path = ''
K = None
z_range = [0.2, 1.1]
local_regions = False
filter_grasps = False
skip_border_objects = False
forward_passes = 5
segmap_id = 0
arg_configs = []

inference(
    global_config,
    ckpt_dir,
    np_path if not png_path else png_path,
    z_range=eval(str(z_range)),
    K=K,
    local_regions=local_regions,
    filter_grasps=filter_grasps,
    segmap_id=segmap_id,
    forward_passes=forward_passes,
    skip_border_objects=skip_border_objects
)