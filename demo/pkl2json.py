from cv2box import CVImage, CVFile
from cv2box.utils.util import get_path_by_ext
from tqdm import tqdm

result_dict = {'data': []}
for pkl_p in tqdm(get_path_by_ext('/ljt/workspace/eft/eft_out/test_04-21_dance_0420_eft_cocon3d/datasets/dy_dance_0420', ext_list=['.pkl'])):
    pkl_data = CVFile(pkl_p).data

    # # check image file name
    # print(str(pkl_data['imageName'][0]))
    # raise '111'

    param_dict = {'parm_pose': pkl_data['pred_pose_rotmat'][0].tolist(),
                  'parm_shape': pkl_data['pred_shape'][0].tolist(), 'bbox_scale': float(pkl_data['scale'][0]),
                  'bbox_center': pkl_data['center'][0].tolist(), 'gt_keypoint_2d': pkl_data['keypoint2d'][0].tolist(),
                  'imageName': str(pkl_data['imageName'][0][58:])}
    result_dict['data'].append(param_dict)

CVFile('/ljt/workspace/eft/eft_fit/dance_0420.json').json_write(result_dict)
