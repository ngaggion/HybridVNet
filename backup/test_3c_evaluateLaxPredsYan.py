import os 
from utils.file_utils import load_folder
import pandas as pd
import SimpleITK as sitk
from medpy.metric import dc
import numpy as np
import cv2

from functools import partial
from multiprocessing import Pool

def MCD(seg_A, seg_B):
    table_md = []
    seg_A = seg_A.transpose(2,1,0)
    seg_B = seg_B.transpose(2,1,0)
    seg_A[seg_A>0] = 1
    seg_B[seg_B>0] = 1
    
    X, Y, Z = seg_A.shape
    for z in range(Z):
        # Binary mask at this slice
        slice_A = seg_A[:, :, z].astype(np.uint8)
        slice_B = seg_B[:, :, z].astype(np.uint8)

        if np.sum(slice_A) > 0 and np.sum(slice_B) > 0:
            contours, _ = cv2.findContours(cv2.inRange(slice_A, 1, 1),
                                              cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_NONE)
            pts_A = contours[0]
            for i in range(1, len(contours)):
                pts_A = np.vstack((pts_A, contours[i]))

            contours, _ = cv2.findContours(cv2.inRange(slice_B, 1, 1),
                                              cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_NONE)
            pts_B = contours[0]
            for i in range(1, len(contours)):
                pts_B = np.vstack((pts_B, contours[i]))
            # Distance matrix between point sets
            M = np.zeros((len(pts_A), len(pts_B)))
            for i in range(len(pts_A)):
                for j in range(len(pts_B)):
                    M[i, j] = np.linalg.norm(pts_A[i, 0] - pts_B[j, 0])

     
            md = 0.5 * (np.mean(np.min(M, axis=0)) + np.mean(np.min(M, axis=1))) * 1.8
            table_md += [md]

    mean_md = np.mean(table_md) if table_md else None
    return mean_md



def HD(seg_A, seg_B):
    
    table_hd = []
    seg_A = seg_A.transpose(2,1,0)
    seg_B = seg_B.transpose(2,1,0)
    seg_A[seg_A>0] = 1
    seg_B[seg_B>0] = 1
    
    
    X, Y, Z = seg_A.shape
    for z in range(Z):
        # Binary mask at this slice
        slice_A = seg_A[:, :, z].astype(np.uint8)
        slice_B = seg_B[:, :, z].astype(np.uint8)

        if np.sum(slice_A) > 0 and np.sum(slice_B) > 0:
            contours, _ = cv2.findContours(cv2.inRange(slice_A, 1, 1),
                                              cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_NONE)
            pts_A = contours[0]
            for i in range(1, len(contours)):
                pts_A = np.vstack((pts_A, contours[i]))

            contours, _ = cv2.findContours(cv2.inRange(slice_B, 1, 1),
                                              cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_NONE)
            pts_B = contours[0]
            for i in range(1, len(contours)):
                pts_B = np.vstack((pts_B, contours[i]))

            # Distance matrix between point sets
            M = np.zeros((len(pts_A), len(pts_B)))
            for i in range(len(pts_A)):
                for j in range(len(pts_B)):
                    M[i, j] = np.linalg.norm(pts_A[i, 0] - pts_B[j, 0])

            hd = np.max([np.max(np.min(M, axis=0)), np.max(np.min(M, axis=1))]) * 1.8
            table_hd += [hd]

    mean_hd = np.mean(table_hd) if table_hd else None
    return mean_hd


def eval(model_path):
    print("Evaluating model", model_path.split("/")[-1])

    dataframe = pd.DataFrame(columns=["ID", 
                            "LA 2CH - DC", "LA 2CH - HD", "LA 2CH - MCD",
                            "LA 4CH - DC", "LA 4CH - HD", "LA 4CH - MCD",
                            "RA 4CH - DC", "RA 4CH - HD", "RA 4CH - MCD"])
    
    i = 0
    masks_path = os.path.join(model_path, "LaxMasks")
    subjects = load_folder(masks_path)
    
    for subject in subjects:
        timesteps = load_folder(subject)
        for time in timesteps:
            mask_path = os.path.join(time, "2CH.nii.gz")
            pred2ch = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
            mask_path = os.path.join(time, "4CH.nii.gz")
            pred4ch = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
            
            try: 
                gt_2CH_path = "../Dataset/LaxMasksYan/mask_2CH/" + subject.split('/')[-1] + "_" + time.split('/')[-1] + "_gt.nii.gz"
                gt_2CH = sitk.GetArrayFromImage(sitk.ReadImage(gt_2CH_path))
                gt_4CH_path = "../Dataset/LaxMasksYan/mask_4CH/" + subject.split('/')[-1] + "_" + time.split('/')[-1] + "_gt.nii.gz"
                gt_4CH = sitk.GetArrayFromImage(sitk.ReadImage(gt_4CH_path))
            except:
                continue
                
            dice_la2ch = dc(gt_2CH == 50, pred2ch == 50)
            hausdorff_la2ch = HD(gt_2CH == 50, pred2ch == 50)
            assd_value_la2ch = MCD(gt_2CH == 50, pred2ch == 50)
            
            dice_la4ch = dc(gt_4CH == 50, pred4ch == 50)
            hausdorff_la4ch = HD(gt_4CH == 50, pred4ch == 50)
            assd_value_la4ch = MCD(gt_4CH == 50, pred4ch == 50)

            dice_ra4ch = dc(gt_4CH == 100, pred4ch == 100)
            hausdorff_ra4ch = HD(gt_4CH == 100, pred4ch == 100)
            assd_value_ra4ch = MCD(gt_4CH == 100, pred4ch == 100)

            subject_id = subject.split('/')[-1]
            print(dice_la2ch,dice_la4ch,dice_ra4ch)
            print(hausdorff_la2ch,hausdorff_la4ch,hausdorff_ra4ch)
            print(assd_value_la2ch,assd_value_la4ch,assd_value_ra4ch)
            
            print("")
            
            dataframe.loc[i] = [subject_id, 
                    dice_la2ch, hausdorff_la2ch, assd_value_la2ch, 
                    dice_la4ch, hausdorff_la4ch, assd_value_la4ch, 
                    dice_ra4ch, hausdorff_ra4ch, assd_value_ra4ch]
            
            i += 1
    
    print("Saving metrics", model_path.split("/")[-1])
    dataframe.to_csv(os.path.join(model_path, "lax_metrics_yan.csv"), index=False)

    return

def evaluate_model(models_path):    
    models = load_folder(models_path)
    
    func = partial(eval)
    with Pool(4) as p:
        p.map(func, models)
        

if __name__ == "__main__":
    models_path = "../Predictions/Surface/"
    evaluate_model(models_path)