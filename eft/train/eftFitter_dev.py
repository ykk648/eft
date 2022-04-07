import torch
import torch.nn as nn
import cv2
from tqdm import tqdm
import os
import datetime
import pickle
from pathlib import Path

from eft.datasets import MixedDataset, BaseDataset
from eft.models import hmr, SMPL  # , SMPLX
from eft.cores import config
from eft.cores import constants
from .fits_dict import FitsDict
from renderer import viewer2D
from renderer import glViewer
from eft.utils.timer import Timer
from eft.utils.data_loader import CheckpointDataLoader
from eft.utils.imutils import deNormalizeBatchImg
from eft.utils.geometry import weakProjection_gpu
from eft.train import Trainer, normalize_2dvector
import eft.utils.smpl_utils as smpl_utils  # import visSMPLoutput, getSMPLoutput, getSMPLoutput_imgspace

g_timer = Timer()


class EFTFitter(Trainer):
    def init_fn(self):
        self.train_ds = MixedDataset(self.options, ignore_3d=self.options.ignore_3d, is_train=True)

        self.model = hmr(config.SMPL_MEAN_PARAMS, pretrained=True).to(self.device)

        if self.options.bExemplarMode:
            # lr = 1e-5   #5e-5 * 0.2       #original
            lr = self.options.lr_eft  # 5e-6       #New EFT
        else:
            lr = self.options.lr

        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr, weight_decay=0)

        self.smpl = SMPL(config.SMPL_MODEL_DIR, batch_size=self.options.batch_size, create_transl=False).to(self.device)

        # Per-vertex loss on the shape
        self.criterion_shape = nn.L1Loss().to(self.device)
        # Keypoint (2D and 3D) loss
        # No reduction because confidence weighting needs to be applied
        self.criterion_keypoints = nn.MSELoss(reduction='none').to(self.device)
        # Loss for SMPL parameter regression
        self.criterion_regr = nn.MSELoss().to(self.device)
        self.models_dict = {'model': self.model}
        self.optimizers_dict = {'optimizer': self.optimizer}
        self.focal_length = constants.FOCAL_LENGTH

        if self.options.pretrained_checkpoint is not None:
            print(">>> Load Pretrained mode: {}".format(self.options.pretrained_checkpoint))
            self.load_pretrained(checkpoint_file=self.options.pretrained_checkpoint)
            self.backupModel()

        # This should be called here after loading model
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = torch.nn.DataParallel(self.model)  # Failed...

        # Load dictionary of fits
        self.fits_dict = FitsDict(self.options, self.train_ds)

        # Create renderer
        self.renderer = None  # Renderer(focal_length=self.focal_length, img_res=self.options.img_res, faces=self.smpl.faces)

        # debug
        from torchvision.transforms import Normalize
        self.de_normalize_img = Normalize(mean=[-constants.IMG_NORM_MEAN[0] / constants.IMG_NORM_STD[0],
                                                -constants.IMG_NORM_MEAN[1] / constants.IMG_NORM_STD[1],
                                                -constants.IMG_NORM_MEAN[2] / constants.IMG_NORM_STD[2]],
                                          std=[1 / constants.IMG_NORM_STD[0], 1 / constants.IMG_NORM_STD[1],
                                               1 / constants.IMG_NORM_STD[2]])

    def exemplerTrainingMode(self):
        for module in self.model.modules():
            if type(module) == False:
                continue

            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                # print(module)
                module.eval()
                for m in module.parameters():
                    m.requires_grad = False
            if isinstance(module, nn.Dropout):
                # print(module)
                module.eval()
                for m in module.parameters():
                    m.requires_grad = False

    # Given a batch, run HMR training
    # Assumes a single sample in the batch, requiring batch norm disabled
    # Loss is a bit different from original HMR training
    def run_eft_step(self, input_batch, iterIdx=0):

        self.model.train()

        if self.options.bExemplarMode:
            self.exemplerTrainingMode()

        # Get data from the batch
        images = input_batch['img']  # input image
        gt_keypoints_2d = input_batch['keypoints'].clone()  # 2D keypoints           #[N,49,3]

        indices = input_batch['sample_index']  # index of example inside its dataset
        batch_size = images.shape[0]

        index_cpu = indices.cpu()
        if self.options.bExemplar_dataLoaderStart >= 0:
            index_cpu += self.options.bExemplar_dataLoaderStart  # Bug fixed.

        # De-normalize 2D keypoints from [-1,1] to pixel space
        gt_keypoints_2d_orig = gt_keypoints_2d.clone()
        gt_keypoints_2d_orig[:, :, :-1] = 0.5 * self.options.img_res * (
                gt_keypoints_2d_orig[:, :, :-1] + 1)  # 49: (25+24) x 3

        # Feed images in the network to predict camera and SMPL parameters
        pred_rotmat, pred_betas, pred_camera = self.model(images)

        pred_output = self.smpl(betas=pred_betas, body_pose=pred_rotmat[:, 1:],
                                global_orient=pred_rotmat[:, 0].unsqueeze(1), pose2rot=False)
        # pred_vertices = pred_output.vertices
        pred_joints_3d = pred_output.joints

        # weakProjection_gpu################
        pred_keypoints_2d = weakProjection_gpu(pred_joints_3d, pred_camera[:, 0], pred_camera[:, 1:])  # N, 49, 2

        if True:  # Ignore hips and hip centers, foot
            LENGTH_THRESHOLD = 0.0089  # 1/112.0     #at least it should be 5 pixel

            # Disable Hips by default
            if self.options.eft_withHip2D == False:
                gt_keypoints_2d[:, 2 + 25, 2] = 0
                gt_keypoints_2d[:, 3 + 25, 2] = 0
                gt_keypoints_2d[:, 14 + 25, 2] = 0

            # #Compute angle knee to ankle orientation
            gt_boneOri_leftLeg = gt_keypoints_2d[:, 5 + 25, :2] - gt_keypoints_2d[:, 4 + 25,
                                                                  :2]  # Left lower leg orientation     #(N,2)
            gt_boneOri_leftLeg, leftLegLeng = normalize_2dvector(gt_boneOri_leftLeg)

            if leftLegLeng > LENGTH_THRESHOLD:
                leftLegValidity = gt_keypoints_2d[:, 5 + 25, 2] * gt_keypoints_2d[:, 4 + 25, 2]
                pred_boneOri_leftLeg = pred_keypoints_2d[:, 5 + 25, :2] - pred_keypoints_2d[:, 4 + 25, :2]
                pred_boneOri_leftLeg, _ = normalize_2dvector(pred_boneOri_leftLeg)
                loss_legOri_left = torch.ones(1).to(self.device) - torch.dot(gt_boneOri_leftLeg.view(-1),
                                                                             pred_boneOri_leftLeg.view(-1))
            else:
                loss_legOri_left = torch.zeros(1).to(self.device)
                leftLegValidity = torch.zeros(1).to(self.device)

            gt_boneOri_rightLeg = gt_keypoints_2d[:, 0 + 25, :2] - gt_keypoints_2d[:, 1 + 25,
                                                                   :2]  # Right lower leg orientation
            gt_boneOri_rightLeg, rightLegLeng = normalize_2dvector(gt_boneOri_rightLeg)
            if rightLegLeng > LENGTH_THRESHOLD:

                rightLegValidity = gt_keypoints_2d[:, 0 + 25, 2] * gt_keypoints_2d[:, 1 + 25, 2]
                pred_boneOri_rightLeg = pred_keypoints_2d[:, 0 + 25, :2] - pred_keypoints_2d[:, 1 + 25, :2]
                pred_boneOri_rightLeg, _ = normalize_2dvector(pred_boneOri_rightLeg)
                loss_legOri_right = torch.ones(1).to(self.device) - torch.dot(gt_boneOri_rightLeg.view(-1),
                                                                              pred_boneOri_rightLeg.view(-1))
            else:
                loss_legOri_right = torch.zeros(1).to(self.device)
                rightLegValidity = torch.zeros(1).to(self.device)
            loss_legOri = leftLegValidity * loss_legOri_left + rightLegValidity * loss_legOri_right

            # Disable Foots
            gt_keypoints_2d[:, 5 + 25, 2] = 0  # Left foot
            gt_keypoints_2d[:, 0 + 25, 2] = 0  # Right foot

        # Compute 2D reprojection loss for the keypoints
        loss_keypoints_2d = self.keypoint_loss(pred_keypoints_2d, gt_keypoints_2d,
                                               self.options.openpose_train_weight,
                                               self.options.gt_train_weight)

        loss_keypoints_3d = torch.tensor(0)

        loss_regr_betas_noReject = torch.mean(pred_betas ** 2)

        loss = self.options.keypoint_loss_weight * loss_keypoints_2d + \
               self.options.beta_loss_weight * loss_regr_betas_noReject + \
               ((torch.exp(-pred_camera[:, 0] * 10)) ** 2).mean()

        if True:  # Leg orientation loss
            loss = loss + 0.005 * loss_legOri

        loss *= 60

        # Do backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Save result
        output = {}
        output['pred_pose_rotmat'] = pred_rotmat.detach().cpu().numpy()
        output['pred_shape'] = pred_betas.detach().cpu().numpy()
        output['imageName'] = input_batch['imgname']
        output['scale'] = input_batch['scale'].detach().cpu().numpy()
        output['center'] = input_batch['center'].detach().cpu().numpy()
        output['keypoint2d'] = input_batch['keypoints_original'].detach().cpu().numpy()
        output['sampleIdx'] = input_batch['sample_index'].detach().cpu().numpy()  # To use loader directly

        losses = {'loss': loss.detach().item(),
                  'loss_keypoints': loss_keypoints_2d.detach().item(),
                  'loss_keypoints_3d': loss_keypoints_3d.detach().item(),
                  #   'loss_regr_pose': loss_regr_pose.detach().item(),
                  'loss_regr_betas': loss_regr_betas_noReject.detach().item()}

        if self.options.bDebug_visEFT:  # g_debugVisualize:    #Debug Visualize input

            # print("Image Name: {}".format(output['imageName']))
            for b in range(batch_size):

                # DenormalizeImg
                curImgVis = deNormalizeBatchImg(images[b].cpu())
                viewer2D.ImShow(curImgVis, name='rawIm', scale=4.0)

                # Visualize GT 2D keypoints
                if True:
                    gt_keypoints_2d_orig_vis = gt_keypoints_2d_orig.detach().cpu().numpy()
                    gt_keypoints_2d_orig_vis[b, :25, 2] = 0  # Don't show openpose
                    curImgVis = viewer2D.Vis_Skeleton_2D_SPIN49(gt_keypoints_2d_orig_vis[b, :, :2],
                                                                gt_keypoints_2d_orig_vis[b, :, 2], bVis=False,
                                                                image=curImgVis)

                # Visualize SMPL in image space
                pred_smpl_output, pred_smpl_output_bbox = smpl_utils.visSMPLoutput_bboxSpace(self.smpl, {
                    "pred_rotmat": pred_rotmat, "pred_shape": pred_betas, "pred_camera": pred_camera}
                                                                                             , image=curImgVis,
                                                                                             waittime=-1)

                # Visualize GT Mesh
                if False:
                    gtOut = {"pred_pose": gt_pose, "pred_shape": gt_betas, "pred_camera": pred_camera}
                    # _, gt_smpl_output_bbox = smpl_utils.getSMPLoutput_bboxSpace(self.smpl, gtOut)
                    _, gt_smpl_output_bbox = smpl_utils.getSMPLoutput_bboxSpace(self.smpl_male,
                                                                                gtOut)  # Assuming Male model
                    gt_smpl_output_bbox['body_mesh']['color'] = glViewer.g_colorSet['hand']
                    glViewer.addMeshData([gt_smpl_output_bbox['body_mesh']], bComputeNormal=True)

                ############### Visualize Skeletons ############### 
                glViewer.setSkeleton([pred_smpl_output_bbox['body_joints_vis']])

                if True:
                    glViewer.show(1)
                elif False:  # Render to Files in original image space

                    # Get Skeletons
                    img_original = cv2.imread(input_batch['imgname'][0])
                    # viewer2D.ImShow(img_original, waitTime=0)
                    bboxCenter = input_batch['center'].detach().cpu()[0]
                    bboxScale = input_batch['scale'].detach().cpu()[0]
                    imgShape = img_original.shape[:2]
                    smpl_output, smpl_output_bbox, smpl_output_imgspace = smpl_utils.getSMPLoutput_imgSpace(self.smpl, {
                        "pred_rotmat": pred_rotmat, "pred_shape": pred_betas, "pred_camera": pred_camera},
                                                                                                            bboxCenter,
                                                                                                            bboxScale,
                                                                                                            imgShape)

                    glViewer.setBackgroundTexture(img_original)  # Vis raw video as background
                    glViewer.setWindowSize(img_original.shape[1] * 2,
                                           img_original.shape[0] * 2)  # Vis raw video as background
                    glViewer.setMeshData([smpl_output_imgspace['body_mesh']],
                                         bComputeNormal=True)  # Vis raw video as background
                    glViewer.setSkeleton([])

                    imgname = os.path.basename(input_batch['imgname'][0])[:-4]
                    fileName = "{0}_{1}_{2:04d}".format(dataset_name[0], imgname, iterIdx)

                    # rawImg = cv2.putText(rawImg,data['subjectId'],(100,100), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,0),2)
                    glViewer.render_on_image('/home/hjoo/temp/render_eft', fileName, img_original, scaleFactor=2)

                else:
                    # Render
                    if True:
                        imgname = output['imageName'][b]
                        root_imgname = os.path.basename(imgname)[:-4]
                        renderRoot = f'/home/hjoo/temp/render_eft/eft_{root_imgname}'
                        imgname = '{:04d}'.format(iterIdx)

                        # smpl_utils.renderSMPLoutput(renderRoot,'overlaid','raw',imgname=imgname)
                        smpl_utils.renderSMPLoutput(renderRoot, 'overlaid', 'mesh', imgname=imgname)
                        smpl_utils.renderSMPLoutput(renderRoot, 'overlaid', 'skeleton', imgname=imgname)
                        smpl_utils.renderSMPLoutput(renderRoot, 'side', 'mesh', imgname=imgname)

        losses['r_error'] = 0

        return output, losses

    # Run EFT
    # Save output as seperate pkl files
    def eftAllInDB(self, eft_out_dir="./eft_out/", bExportPKL=True, test_dataset_h36m=None):

        now = datetime.datetime.now()
        newName = 'test_{:02d}-{:02d}'.format(now.month, now.day)
        outputDir = newName + '_' + self.options.db_set + '_' + self.options.name

        # exemplarOutputPath = os.path.join(config.EXEMPLAR_OUTPUT_ROOT , outputDir)
        os.makedirs(eft_out_dir, exist_ok=True)
        exemplarOutputPath = os.path.join(eft_out_dir, outputDir)
        os.makedirs(exemplarOutputPath, exist_ok=True)

        """Training process."""
        # Run training for num_epochs epochs
        # Create new DataLoader every epoch and (possibly) resume from an arbitrary step inside an epoch
        train_data_loader = CheckpointDataLoader(self.train_ds, checkpoint=self.checkpoint,
                                                 batch_size=1,  # Always o1
                                                 num_workers=self.options.num_workers,
                                                 pin_memory=self.options.pin_memory,
                                                 shuffle=False)  # No Shuffle

        maxExemplarIter = self.options.maxExemplarIter

        # Iterate over all batches in an epoch
        for step, batch in enumerate(tqdm(train_data_loader)):  # , desc='Epoch '+str(epoch),

            # bSkipExisting = self.options.bNotSkipExemplar == False  # bNotSkipExemplar ===True --> bSkipExisting==False
            # if bSkipExisting:
            #     fileNameOnly = os.path.basename(batch['imgname'][0])[:-4]
            #     sampleIdx = batch['sample_index'][0].item()
            #     if self.options.bExemplar_dataLoaderStart >= 0:
            #         sampleIdx += self.options.bExemplar_dataLoaderStart
            #     fileName = '{}_{}.pkl'.format(fileNameOnly, sampleIdx)
            #     outputPath = os.path.join(exemplarOutputPath, fileName)
            #     if os.path.exists(outputPath):
            #         print("Skipped: {}".format(outputPath))
            #         continue

            # g_timer.tic()
            self.reloadModel()  # For each sample

            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            for it in range(maxExemplarIter):
                output, losses = self.run_eft_step(batch, iterIdx=it)

                output['loss_keypoints_2d'] = losses['loss_keypoints']
                # Thresholding by 2D keypoint error
                if True:
                    if output['loss_keypoints_2d'] < self.options.eft_thresh_keyptErr_2d:  # 1e-4:
                        # glViewer.show(0)
                        break

            # g_timer.toc(average=True, bPrint=True, title="wholeEFT")

            # glViewer.show(0)

            if bExportPKL:  # Export Output to PKL files

                fileNameOnly = output['imageName'][0][31:-4]

                sampleIdx = output['sampleIdx'][0].item()
                if self.options.bExemplar_dataLoaderStart >= 0:
                    sampleIdx += self.options.bExemplar_dataLoaderStart

                fileName = '{}.pkl'.format(fileNameOnly)
                outputPath = exemplarOutputPath + fileName
                os.makedirs(str(Path(outputPath).parent), exist_ok=True)

                # print("Saved:{}".format(outputPath))
                with open(outputPath, 'wb') as f:
                    pickle.dump(output, f)
                    f.close()
