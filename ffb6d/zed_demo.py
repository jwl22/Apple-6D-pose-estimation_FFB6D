#!/usr/bin/env python3
from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import sys
import pyzed.sl as sl
import time
import os
import tqdm
import cv2
import torch
import argparse
import torch.nn as nn
import numpy as np
import pickle as pkl
from common import Config, ConfigRandLA
from models.ffb6d import FFB6D
from datasets.ycb.zed_ycb_dataset import Dataset as YCB_Dataset

# from datasets.ycb.ycb_dataset import Dataset as YCB_Dataset
from datasets.linemod.linemod_dataset import Dataset as LM_Dataset
from utils.pvn3d_eval_utils_kpls import cal_frame_poses, cal_frame_poses_lm
from utils.basic_utils import Basic_Utils

try:
    from neupeak.utils.webcv2 import imshow, waitKey
except ImportError:
    from cv2 import imshow, waitKey


parser = argparse.ArgumentParser(description="Arg parser")
parser.add_argument(
    "-checkpoint",
    type=str,
    default="train_log/ycb/checkpoints/FFB6D_best.pth.tar",
    help="Checkpoint to eval",
)
parser.add_argument(
    "-dataset",
    type=str,
    default="ycb",
    help="Target dataset, ycb or linemod. (linemod as default).",
)
parser.add_argument(
    "-cls",
    type=str,
    default="ape",
    help="Target object to eval in LineMOD dataset. (ape, benchvise, cam, can,"
    + "cat, driller, duck, eggbox, glue, holepuncher, iron, lamp, phone)",
)
parser.add_argument("-show", action="store_true", help="View from imshow or not.")
args = parser.parse_args()

if args.dataset == "ycb":
    config = Config(ds_name=args.dataset)
else:
    config = Config(ds_name=args.dataset, cls_type=args.cls)
bs_utils = Basic_Utils(config)

## ZED camera setting
zed = sl.Camera()
# initialize camera parameters
init_params = sl.InitParameters()
# init_params.set_from_svo_file('Your File')

init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # Use Ultra depth mode
init_params.coordinate_units = sl.UNIT.METER
init_params.depth_minimum_distance = 0.3
init_params.depth_maximum_distance = 2.0
# init_params.camera_resolution = sl.RESOLUTION.AUTO
init_params.camera_resolution = sl.RESOLUTION.VGA
init_params.camera_fps = 30

rt_param = sl.RuntimeParameters()
# rt_param.sensing_mode = sl.SENSING_MODE.FILL
rt_param.enable_fill_mode = True

# Create ZED object

zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, 16)
# Set exposure to 50% of camera framerate
# Set camera settings for better brightness
zed.set_camera_settings(
    sl.VIDEO_SETTINGS.BRIGHTNESS, 4
)  # Adjust brightness level (0 to 8)
zed.set_camera_settings(sl.VIDEO_SETTINGS.CONTRAST, 4)  # Adjust contrast level (0 to 8)
zed.set_camera_settings(sl.VIDEO_SETTINGS.GAMMA, 5)
zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, -1)
# Set white balance to 4600K
zed.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE, 0)
# Reset to auto exposure
# zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, -1)
zed.set_camera_settings(sl.VIDEO_SETTINGS.HUE, 0)
zed.set_camera_settings(sl.VIDEO_SETTINGS.SATURATION, 4)

# zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE_TIME, 100)
# Open the SVO file specified as a parameter
err = zed.open(init_params)
# if the video doesn't open successfully
if err != sl.ERROR_CODE.SUCCESS:
    sys.stdout.write(repr(err))
    zed.close()
    exit()
# Prepare container
Depth_image = sl.Mat()
Left_image = sl.Mat()
nb_frames = zed.get_svo_number_of_frames()


def ensure_fd(fd):
    if not os.path.exists(fd):
        os.system("mkdir -p {}".format(fd))


def load_checkpoint(model=None, optimizer=None, filename="checkpoint"):
    filename = "{}.pth.tar".format(filename)

    assert os.path.isfile(filename), "==> Checkpoint '{}' not found".format(filename)
    print("==> Loading from checkpoint '{}'".format(filename))
    try:
        checkpoint = torch.load(filename)
    except Exception:
        checkpoint = pkl.load(open(filename, "rb"))
    epoch = checkpoint.get("epoch", 0)
    it = checkpoint.get("it", 0.0)
    best_prec = checkpoint.get("best_prec", None)
    if model is not None and checkpoint["model_state"] is not None:
        ck_st = checkpoint["model_state"]
        if "module" in list(ck_st.keys())[0]:
            tmp_ck_st = {}
            for k, v in ck_st.items():
                tmp_ck_st[k.replace("module.", "")] = v
            ck_st = tmp_ck_st
        model.load_state_dict(ck_st)
    if optimizer is not None and checkpoint["optimizer_state"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    print("==> Done")
    return it, epoch, best_prec


def cal_view_pred_pose(model, data, epoch=0, obj_id=-1):
    result_6d = None
    model.eval()
    with torch.set_grad_enabled(False):
        cu_dt = {}
        # device = torch.device('cuda:{}'.format(args.local_rank))
        for key in data.keys():
            # if data[key].dtype in [np.float32, np.uint8]:
            #     cu_dt[key] = torch.from_numpy(data[key].astype(np.float32)).cuda()
            # elif data[key].dtype in [np.int32, np.uint32]:
            #     cu_dt[key] = torch.LongTensor(data[key].astype(np.int32)).cuda()
            # elif data[key].dtype in [torch.uint8, torch.float32]:
            #     cu_dt[key] = data[key].float().cuda()
            # elif data[key].dtype in [torch.int32, torch.int16]:
            #     cu_dt[key] = data[key].long().cuda()

            if data[key].dtype in [np.float32, np.uint8]:
                # cu_dt[key] = torch.Tensor([data[key]]).type(torch.float32).cuda()
                cu_dt[key] = torch.Tensor(
                    np.array([data[key]]).astype(np.float32)
                ).cuda()
            elif data[key].dtype in [np.int32, np.uint32]:
                # cu_dt[key] = torch.LongTensor([data[key]]).type(torch.int32).cuda()
                cu_dt[key] = torch.LongTensor(
                    np.array([data[key]]).astype(np.int32)
                ).cuda()
            # 아래 if문은 실행되지 않음
            elif data[key].dtype in [torch.uint8, torch.float32]:
                cu_dt[key] = data[key].type(torch.float32).cuda()
            elif data[key].dtype in [torch.int32, torch.int16]:
                cu_dt[key] = data[key].type(torch.int32).cuda()
        end_points = model(cu_dt)
        _, classes_rgbd = torch.max(end_points["pred_rgbd_segs"], 1)

        pcld = cu_dt["cld_rgb_nrm"][:, :3, :].permute(0, 2, 1).contiguous()
        if args.dataset == "ycb":
            pred_cls_ids, pred_pose_lst, _ = cal_frame_poses(
                pcld[0],
                classes_rgbd[0],
                end_points["pred_ctr_ofs"][0],
                end_points["pred_kp_ofs"][0],
                True,
                config.n_objects,
                True,
                None,
                None,
            )
        else:
            pred_pose_lst = cal_frame_poses_lm(
                pcld[0],
                classes_rgbd[0],
                end_points["pred_ctr_ofs"][0],
                end_points["pred_kp_ofs"][0],
                True,
                config.n_objects,
                False,
                obj_id,
            )
            pred_cls_ids = np.array([[1]])

        np_rgb = cu_dt["rgb"].cpu().numpy().astype("uint8")[0].transpose(1, 2, 0).copy()
        if args.dataset == "ycb":
            np_rgb = np_rgb[:, :, ::-1].copy()
        ori_rgb = np_rgb.copy()
        for cls_id in cu_dt["cls_ids"][0].cpu().numpy():
            idx = np.where(pred_cls_ids == cls_id)[0]
            if len(idx) == 0:
                continue
            pose = pred_pose_lst[idx[0]]
            result_6d = pose
            if args.dataset == "ycb":
                obj_id = int(cls_id[0])
            mesh_pts = bs_utils.get_pointxyz(obj_id, ds_type=args.dataset).copy()
            mesh_pts = np.dot(mesh_pts, pose[:, :3].T) + pose[:, 3]
            if args.dataset == "ycb":
                K = config.intrinsic_matrix["ycb_K1"]
            else:
                K = config.intrinsic_matrix["linemod"]
            mesh_p2ds = bs_utils.project_p3d(mesh_pts, 1.0, K)
            color = bs_utils.get_label_color(obj_id, n_obj=6, mode=2)
            np_rgb = bs_utils.draw_p2ds(np_rgb, mesh_p2ds, color=color)
        vis_dir = os.path.join(config.log_eval_dir, "pose_vis")
        ensure_fd(vis_dir)
        f_pth = os.path.join(vis_dir, "{}.jpg".format(epoch))
        if args.dataset == "ycb":
            bgr = np_rgb
            ori_bgr = ori_rgb
        else:
            bgr = np_rgb[:, :, ::-1]
            ori_bgr = ori_rgb[:, :, ::-1]
        # cv2.imwrite(f_pth, bgr)
        if args.show:
            imshow("projected_pose_rgb", bgr)
            imshow("original_rgb", ori_bgr)
            waitKey()
    # if epoch == 0:
    # print("\n\nResults saved in {}".format(vis_dir))

    cv2.imshow("Result", bgr)
    cv2.waitKey(1)
    # if cv2.waitKey() == ord("q"):
    #     cv2.destroyAllWindows()
    #     zed.close()
    #       exit()
    return result_6d


def main():
    if args.dataset == "ycb":
        test_ds = YCB_Dataset("test")
        obj_id = -1
    # else:
    #     test_ds = LM_Dataset("test", cls_type=args.cls)
    #     obj_id = config.lm_obj_dict[args.cls]
    # test_loader = torch.utils.data.DataLoader(
    #     test_ds, batch_size=config.test_mini_batch_size, shuffle=False, num_workers=20
    # )

    rndla_cfg = ConfigRandLA
    model = FFB6D(
        n_classes=config.n_objects,
        n_pts=config.n_sample_points,
        rndla_cfg=rndla_cfg,
        n_kps=config.n_keypoints,
    )
    model.cuda()

    # load status from checkpoint
    if args.checkpoint is not None:
        load_checkpoint(model, None, filename=args.checkpoint[:-8])

    i = 0
    while True:
        if zed.grab(rt_param) == sl.ERROR_CODE.SUCCESS:
            # get the frame index
            # svo_position = zed.get_svo_position()

            zed.retrieve_measure(Depth_image, sl.MEASURE.DEPTH)
            depth_image_rgba = Depth_image.get_data()
            depth_image = cv2.cvtColor(depth_image_rgba, cv2.COLOR_RGBA2RGB)
            depth_image *= 255 / (depth_image.max() - depth_image.min())
            depth_image = 255 - depth_image
            cv2.imwrite("tmp_depth.png", depth_image)
            depth_image = cv2.imread("tmp_depth.png", cv2.IMREAD_GRAYSCALE)
            di = depth_image

            zed.retrieve_image(Left_image, sl.VIEW.LEFT)
            image_left = Left_image.get_data()
            image_left = cv2.cvtColor(image_left, cv2.COLOR_RGB2BGR)
            ri = image_left

            # test_loader = torch.utils.data.DataLoader(
            #     test_ds,
            #     batch_size=config.test_mini_batch_size,
            #     shuffle=False,
            # )

            tmp = test_ds.get_item(di, ri)

            # test_loader = {}
            # for key in tmp.keys():
            #     if tmp[key].dtype in [np.float32, np.uint8]:
            #         test_loader[key] = torch.Tensor([tmp[key]]).type(torch.float32)
            #     elif tmp[key].dtype in [np.int32, np.uint32]:
            #         test_loader[key] = torch.LongTensor([tmp[key]]).type(torch.int32)
            #     elif tmp[key].dtype in [torch.uint8, torch.float32]:
            #         test_loader[key] = tmp[key].type(torch.float32)
            #     elif tmp[key].dtype in [torch.int32, torch.int16]:
            #         test_loader[key] = tmp[key].type(torch.int32)

            pose = cal_view_pred_pose(model, tmp, epoch=i, obj_id=obj_id)
            if pose is not None:
                print(pose)

        i += 1

    # for i, data in tqdm.tqdm(enumerate(test_loader), leave=False, desc="val"):
    #     cal_view_pred_pose(model, data, epoch=i, obj_id=obj_id)

    zed.close()


if __name__ == "__main__":
    main()

# vim: ts=4 sw=4 sts=4 expandtab
