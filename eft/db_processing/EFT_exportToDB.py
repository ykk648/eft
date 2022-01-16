import os
from os.path import join
import sys
import json
import numpy as np
# from .read_openpose import read_openpose

import cv2
from renderer import viewer2D
from renderer import glViewer

# from read_openpose import read_openpose

from os import listdir
from os.path import isfile, join
import pickle

sys.path.append('/home/hjoo/codes/SPIN/utils')
sys.path.append('/home/hjoo/codes/SPIN')
from imutils import crop, flip_img, flip_pose, flip_kp, transform, rot_aa
from fairmocap.core import constants 
from fairmocap.core import config 
# from models import hmr, SMPL
from fairmocap.models import hmr, SMPL
import torch
import pickle
from tqdm import tqdm

from torchgeometry import angle_axis_to_rotation_matrix, rotation_matrix_to_angle_axis


def IsValid(data):
    ours_betas = torch.from_numpy(data['pred_shape'])
    if abs(torch.max( abs(ours_betas)).item()) >3:
        return False

    # if data['loss_keypoints_2d']>0.01:
    #     return False

    if 'loss_keypoints_2d_init' in data.keys()  and data['loss_keypoints_2d_init']>0.03:
        print("Rejected: loss_keypoints_2d_init: {}>0.03".format(data['loss_keypoints_2d_init']))
        return False


    if data['loss_keypoints_2d']>0.001:
        print("Rejected: loss_keypoints_2d: {}>0.001".format(data['loss_keypoints_2d']))

        return False
    
    return True




def IsValid_strict(data):
    ours_betas = torch.from_numpy(data['pred_shape'])
    if abs(torch.max( abs(ours_betas)).item()) >1.0:
        return False
    
    if 'loss_keypoints_2d_init' in data.keys()  and data['loss_keypoints_2d_init']>0.04:
        print("Rejected: loss_keypoints_2d_init: {}>0.03".format(data['loss_keypoints_2d_init']))
        return False


    if data['loss_keypoints_2d']>0.0003:
        print("Rejected: loss_keypoints_2d: {}>0.001".format(data['loss_keypoints_2d']))

        return False
    
    return True



if __name__ == '__main__':

    # inputDir = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/mpii_new_wrong_modelloadbug'
    # inputDir = '/home/hjoo/Dropbox (Facebook)/spinExemplar/11-01-46573_lspet'
    
    # inputDir = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/11-01-46497_coco'
    
    # inputDir = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/11-01-52093_lspet_naiveBeta'
    

    

    # inputDir = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/11-03-68285_lspet_ex_lspet_original_naivebeta'
    # inputDir_meta = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/meta_lspet_2dskeletons'

    # inputDir = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/11-01-52093_lspet_naiveBeta'
    # inputDir_meta = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/meta_lspet_2dskeletons'


    # inputDir = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/11-01-52066_mpii_naiveBeta'
    # inputDir_meta = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/meta_mpii_2dskeletons'


    # inputDir = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/11-01-51903_coco_naiveBeta'
    # inputDir_meta ='/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/meta_coco_2dskeletons'

    
    # inputDir = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/11-04_lspet_originalCode_weak_meta'
    # inputDir = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/11-04_mpii_originalCode_weak_meta'
    # inputDir_meta = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/meta_lspet_2dskeletons'


    # inputDir = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/11-04_coco_originalCode_weak_meta'
    # inputDir = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/11-04_pennaction_originalCode_weak_meta'
    # inputDir = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/11-05_panoptic_initFit'
    inputDir = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/11-05_panoptic_initFit_re'

    inputDir = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/11-05_mpii_kneePrior_0.001'
    inputDir = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/11-05_coco_kneePrior_0.001'

    inputDir = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/11-05_mpii_noHipFoot'

    inputDir = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/11-05_coco_noHipFoot'

    inputDir = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/11-05_lspet_noHipFoot'

    
    

    inputDir = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/11-06_mpii_legOriLoss'
    inputDir = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/11-06_coco_legOriLoss'
    inputDir = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/11-06_lspet_legOriLoss'

    
    inputDir ='/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/11-08_lspet_with8143'
    inputDir ='/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/11-08_cocoplus_with8143'
    inputDir = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/11-08_posetrack_with8143'



    inputDir = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/11-13_cocoplus3d_analysis_50'


    inputDir = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/11-13_mpii_analysis_50'
    inputDir = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/11-13_coco_analysis_50'
    
    inputDir = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/11-13_posetrack3d_analysis_50'

    inputDir = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/11-05_panoptic_refit'

    inputDir = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/11-14_lsp_analysis_100'

    inputDir = '/run/media/hjoo/disk/data/cvpr2020_eft_researchoutput/0_SPIN/0_exemplarOutput/04-23_panoptic_with8143_iter60'

    inputDir = '/run/media/hjoo/disk/data/cvpr2020_eft_researchoutput/0_SPIN/0_exemplarOutput/01-28_ochuman_with8143_annotId'


    inputDir = '/run/media/hjoo/disk/data/cvpr2020_eft_researchoutput/0_SPIN/0_smplify/05-18_coco_smplify_withSpin'

    inputDir = '/run/media/hjoo/disk/data/cvpr2020_eft_researchoutput/0_SPIN/0_exemplarOutput/05-25_cocoall_with1336_iterThr'
   
    inputDir = '/run/media/hjoo/disk/data/cvpr2020_eft_researchoutput/0_SPIN/0_smplify/05-25_coco_smplify_fromSpin'

    inputDir = '/run/media/hjoo/disk/data/cvpr2020_eft_researchoutput/0_SPIN/0_exemplarOutput/05-18_cocoall_with1336'
    inputDir ='/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/11-08_coco_with8143'


    inputDir = '/run/media/hjoo/disk/data/cvpr2020_eft_researchoutput/0_SPIN/0_exemplarOutput/01-28_ochuman_with8143_annotId'

    inputDir ='/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/11-08_mpii_with8143'
    inputDir = '/run/media/hjoo/disk/data/cvpr2020_eft_researchoutput/0_SPIN/0_exemplarOutput/01-28_ochuman_with8143_annotId'
    inputDir = '/run/media/hjoo/disk/data/cvpr2020_eft_researchoutput/0_SPIN/0_exemplarOutput/05-25_cocoall_with1336_iterThr'
    inputDir ='/run/media/hjoo/disk/data/cvpr2020_eft_researchoutput/0_SPIN/0_exemplarOutput/09-02_posetrack_eft_with8653_nipsbest'

    # "--img_dir","/run/media/hjoo/disk/data/posetrack","--fit_dir","/run/media/hjoo/disk/data/cvpr2020_eft_researchoutput/0_SPIN/0_exemplarOutput/09-02_posetrack_eft_with8653_nipsbest",

    dbName=None
    if 'mpii' in os.path.basename(inputDir):
        imgDir = '/run/media/hjoo/disk/data/mpii_human_pose_v1/images'
        dbName = 'mpii'
    elif 'pennaction' in os.path.basename(inputDir):
        imgDir = '/run/media/hjoo/disk/data/Penn_Action/frames'
        dbName = 'pennaction'
    elif 'lsp_' in os.path.basename(inputDir):
        imgDir = '/run/media/hjoo/disk/data/lsp_dataset_original'
        dbName = 'lsp'
    elif 'lspet' in os.path.basename(inputDir):
        imgDir = '/run/media/hjoo/disk/data/lspet_dataset/images_highres'
        dbName = 'lspet'
    elif '_cocoplus' in os.path.basename(inputDir):
        imgDir = '/run/media/hjoo/disk/data/coco/train2014'
        dbName = 'cocoplus'
    # elif '_coco' in os.path.basename(inputDir):
    #     imgDir = '/run/media/hjoo/disk/data/coco'
    #     dbName = 'coco'

    elif '_coco' in os.path.basename(inputDir):
        imgDir = '/run/media/hjoo/disk/data/coco2017/annotations/panoptic_train2017_semmap'
        dbName = 'coco_semmap'

    elif '_panoptic' in os.path.basename(inputDir):
        imgDir = '/run/media/hjoo/disk/data/panoptic_mtc/a4_release/hdImgs'
        dbName = 'panoptic'

    elif '_ochuman' in os.path.basename(inputDir):
        imgDir =  '/run/media/hjoo/disk/data/OCHuman/images'
        dbName = 'ochuman'
    
    elif '_posetrack' in os.path.basename(inputDir):
        imgDir =  '/run/media/hjoo/disk/data/posetrack/images/train'
        dbName = 'posetrack_train'


    else:
        assert False


    assert dbName is not None

    fileList  = listdir(inputDir)
    # fileList  = listdir(inputDir_meta)


    smpl = SMPL(config.SMPL_MODEL_DIR,
                         batch_size=1,
                         create_transl=False)


    # structs we need
    imgnames_, scales_, centers_, parts_, openposes_ = [], [], [], [], []
    #additional 3D
    poses_ , shapes_, skel3D_, has_smpl_  = [], [] ,[], []
    annot_ids_ =[]

    loss_keypoints_2d_list =[]
    maxShape_list =[]
    rejectedNum = 0
    acceptedNum =0
    for ii in tqdm(range(len(fileList))):
        fName = fileList[ii]

        fileFullPath = join(inputDir, fName)
        with open(fileFullPath,'rb') as f:
            dataList = pickle.load(f)

        # fileFullPath_meta = join(inputDir_meta, fName)
        # with open(fileFullPath_meta,'rb') as f:
        #     data_meta = pickle.load(f)
        #     data['keypoint2d'] =  data_meta['keypoint2d']       #(1,25+24,3)

        if 'imageName' in dataList.keys():
            dataList_ba ={}
            dataList_ba['0'] = dataList
            dataList = dataList_ba

        for d in dataList:
            data = dataList[d]
            imgName = data['imageName'][0]


            if 'semmap' in dbName:
                imgName = os.path.join( imgDir, os.path.basename(imgName)[15:-3]+"png")

            if os.path.exists==False:
                print(f"Img doesn't exist: {imgName}")
                assert False

            if True:
                if dbName=='pennaction' or dbName =='posetrack_train':
                # if dbName=='pennaction':
                    imgFullPath =os.path.join(imgDir, os.path.basename(os.path.dirname(imgName)), os.path.basename(imgName) )
                elif dbName=='panoptic':
                        imgFullPath =os.path.join(imgDir, os.path.basename( os.path.dirname(os.path.dirname(imgName))),  os.path.basename(os.path.dirname(imgName)), os.path.basename(imgName) )
                elif dbName=='panoptic':
                        imgFullPath =os.path.join(imgDir, os.path.basename( os.path.dirname(os.path.dirname(imgName))),  os.path.basename(os.path.dirname(imgName)), os.path.basename(imgName) )
                else:
                    # imgFullPath =os.path.join(imgDir, os.path.basename(imgFullPath) )
                    imgFullPath =os.path.join(imgDir, os.path.basename(imgName) )
            scale = data['scale'][0]
            center = data['center'][0]


            ours_betas = torch.from_numpy(data['pred_shape'])
            if True:
                loss_keypoints_2d_list.append(data['loss_keypoints_2d'])
                maxShape_list.append( np.max(abs(data['pred_shape'])))
                # continue

           


            # if dbName != 'panoptic':
            #     if IsValid(data)== False:# abs(torch.max( abs(ours_betas)) .item()) >3:
            #         rejectedNum+=1
            #         print("Rejected: so far{}/{}".format(rejectedNum, acceptedNum))
            #         continue

            #     if dbName =='mpii':
            #         if IsValid_strict(data)== False:# abs(torch.max( abs(ours_betas)) .item()) >3:
            #             rejectedNum+=1
            #             print("Rejected: so far{}/{}".format(rejectedNum, acceptedNum))
            #             continue
            # else:
            #     ours_betas = torch.from_numpy(data['pred_shape'])
            #     if (abs(torch.max( abs(ours_betas)).item()) >5 or data['loss_keypoints_2d']>0.01):
            #         rejectedNum+=1
            #         print("Rejected: so far{} /{}".format(rejectedNum, acceptedNum))
            #         continue

            bDraw = False
            # if data['test_error_3dpw']>65:
            # # if data['test_error_3dpw']<70 and data['test_error_3dpw']>65:
            #     bDraw = True
            #     # rejectedNum+=1
            #     # print("Rejected: so far{}/{}".format(rejectedNum, acceptedNum))
            #     # continue

            #Filtering 1
            if 'test_error_3dpw' in data and data['test_error_3dpw']>69:
                rejectedNum+=1
                print("Rejected: so far{}/{}".format(rejectedNum, acceptedNum))
                continue

            #Filtering 2        #Check number of valid joint
            limbJointValidty = sum(data['keypoint2d'][0,25:37,2])

            if limbJointValidty<12:        #All limb valid
                rejectedNum+=1
                # print("Rejected: so far{}/{}".format(rejectedNum, acceptedNum))
                continue

            acceptedNum +=1

            if center[0]<0:
                rejectedNum+=1
                print("wrong!!: {} {}".format(center, scale))
                continue

            # print(abs(torch.max( abs(ours_betas)).item()))
            
            ours_pose_rotmat = torch.from_numpy(data['pred_pose_rotmat'])

            if 'opt_beta' in data:
                spin_betas = torch.from_numpy(data['opt_beta'])
                spin_pose = torch.from_numpy(data['opt_pose'])
            else:
                spin_betas = torch.from_numpy(data['spin_beta'])
                spin_pose = torch.from_numpy(data['spin_pose'])

            pred_camera_vis = data['pred_camera']
            keypoint2d_49 = data['keypoint2d']


            pred_rotmat_hom = torch.cat([ours_pose_rotmat.view(-1, 3, 3), torch.tensor([0,0,1], dtype=torch.float32,).view(1, 3, 1).expand(ours_pose_rotmat.shape[0] * 24, -1, -1)], dim=-1)
            ours_pose_aa = rotation_matrix_to_angle_axis(pred_rotmat_hom).contiguous().view(ours_pose_rotmat.shape[0], -1)
            # ours_pose_aa = pred_aa.cpu().numpy()

            if np.isnan(np.max(ours_pose_aa.numpy())):
                print("Warning: !!NAN detected!!!: {}".format(imgName))
                continue


            #Visualize SMPL output
            # Note that gt_model_joints is different from gt_joints as it comes from SMPL
            # ours_output = smpl(betas=ours_betas, body_pose=ours_pose_rotmat[:,1:], global_orient=ours_pose_rotmat[:,0].unsqueeze(1), pose2rot=False )
            ours_output = smpl(betas=ours_betas, body_pose=ours_pose_aa[:,3:], global_orient=ours_pose_aa[:,:3])
            # spin_output = smpl(betas=spin_betas, body_pose=spin_pose[:,3:], global_orient=spin_pose[:,:3])


            ours_joints_3d = ours_output.joints.detach().cpu().numpy() 
            # spin_vertices = spin_output.vertices.detach().cpu().numpy() 
            # spin_joints_3d = /spin_output.joints.detach().cpu().numpy() 

            #Centering
            if False:
                ours_pelvis = (ours_joints_3d[:, 27:28,:3] + ours_joints_3d[:,28:29,:3]) / 2
                ours_joints_3d[:,:,:3] = ours_joints_3d[:,:,:3] - ours_pelvis        #centering
                ours_vertices = ours_vertices - ours_pelvis

                # spin_model_pelvis = (spin_joints_3d[:, 27:28,:3] + spin_joints_3d[:,28:29,:3]) / 2
                # spin_joints_3d[:,:,:3] = spin_joints_3d[:,:,:3] - spin_model_pelvis        #centering
                # spin_vertices = spin_vertices - spin_model_pelvis

            # if ii %100==0:
            #     bDraw =True

            if bDraw:
                    ours_vertices = ours_output.vertices.d().numpy()
                    rawImg = cv2.imread(imgFullPath)
                    croppedImg = crop(rawImg, center, scale, 
                            [constants.IMG_RES, constants.IMG_RES])
                    rawImg = viewer2D.Vis_Skeleton_2D_SPIN49(data['keypoint2d'][0][:,:2], pt2d_visibility= data['keypoint2d'][0][:,2], image=rawImg)


                    #Visualize image
                    viewer2D.ImShow(rawImg, name='rawImg')
                    viewer2D.ImShow(croppedImg, name='croppedImg')
                        
                    b =0
                    ############### Visualize Mesh ############### 
                    pred_vert_vis = ours_vertices[b]
                    pred_vert_vis *=pred_camera_vis[b,0]
                    pred_vert_vis[:,0] += pred_camera_vis[b,1]        #no need +1 (or  112). Rendernig has this offset already
                    pred_vert_vis[:,1] += pred_camera_vis[b,2]        #no need +1 (or  112). Rendernig has this offset already
                    pred_vert_vis*=112
                    pred_meshes = {'ver': pred_vert_vis, 'f': smpl.faces}

                    # pred_vert_vis = spin_vertices[b]
                    # pred_vert_vis *=pred_camera_vis_init[b,0]
                    # pred_vert_vis[:,0] += pred_camera_vis_init[b,1]        #no need +1 (or  112). Rendernig has this offset already
                    # pred_vert_vis[:,1] += pred_camera_vis_init[b,2]        #no need +1 (or  112). Rendernig has this offset already
                    # pred_vert_vis*=112
                    # spin_meshes = {'ver': pred_vert_vis, 'f': smpl.faces}

                    glViewer.setMeshData([pred_meshes], bComputeNormal= True)
                    # glViewer.setMeshData([pred_meshes, spin_meshes], bComputeNormal= True)
                    # glViewer.setMeshData([pred_meshes], bComputeNormal= True)
                    # glViewer.SetMeshColor('red')

                    glViewer.setBackgroundTexture(croppedImg)
                    glViewer.setWindowSize(croppedImg.shape[1], croppedImg.shape[0])
                    glViewer.SetOrthoCamera(True)
                    glViewer.show(0)


            #Save data
            if dbName == 'mpii' or dbName=='lsp':
                imgnames_.append(os.path.join('images',os.path.basename(imgName))) 
            elif dbName == 'coco':
                imgnames_.append(os.path.join('train2014',os.path.basename(imgName))) 

            elif dbName == 'coco_semmap':
                imgnames_.append(os.path.basename(imgName))         #No folder name

            elif dbName =='pennaction' or dbName =='posetrack' or dbName =='posetrack_train':
                imgname_wDir = os.path.join( os.path.basename(os.path.dirname(imgName)), os.path.basename(imgName) )
                imgnames_.append(imgname_wDir)
            elif dbName =='panoptic':
                imgname_wDir = os.path.join(os.path.basename( os.path.dirname(os.path.dirname(imgName))),  os.path.basename(os.path.dirname(imgName)), os.path.basename(imgName) )
                imgnames_.append(imgname_wDir)
            else:
                imgnames_.append(os.path.join(os.path.basename(imgName))) 
            centers_.append(center)
            scales_.append(scale)
            
            has_smpl_.append(1)        
            poses_.append(ours_pose_aa.numpy()[0])        #(72,)
            # shapes_.append(data['opt_beta'][0])       #(10,)
            shapes_.append(data['pred_shape'][0])       #(10,)

            parts_.append(data['keypoint2d'][0,25:,:])
            openposes_.append(data['keypoint2d'][0,:25,:] )       #blank

            S = np.zeros([24,4])
            S[:,:3] = ours_joints_3d[0,25:,:]
            S[:,3] = 1
            skel3D_.append(S)

            if "annotId" in data:
                # print(data['annotId'][0])
                annot_ids_.append(data['annotId'][0])

            # if ii==10:      #Debug
            # break
    if False:
        viewer2D.Plot(loss_keypoints_2d_list)
        viewer2D.Plot(maxShape_list)

    out_path = '0_ours_dbgeneration'
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    
    out_file = os.path.join(out_path, '{}3d_{}.npz'.format( dbName, os.path.basename(inputDir)) )
    print("Saving to: {}".format(out_file))

    np.savez(out_file, imgname=imgnames_,
                       center=centers_,
                       scale=scales_,
                       part=parts_,
                       openpose=openposes_,
                       pose=poses_,

                        shape=shapes_,
                        has_smpl=has_smpl_,
                        annotIds=annot_ids_)
                        # ,
                        # S=skel3D_)        #Do not export S
    
    
    # exportOursToSpin(cocoPose3DAll, '0_ours_dbgeneration')
    # # coco_extract('/run/media/hjoo/disk/data/coco', None, 'coco_3d_train')