import cv2
import numpy as np

def MCD(seg_A, seg_B):
    table_md = []
    seg_A = seg_A[1:len(seg_A)-2, :, :].transpose(2,1,0)
    seg_B = seg_B[1:len(seg_B)-2, :, :].transpose(2,1,0)
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
    seg_A = seg_A[1:len(seg_A)-2, :, :].transpose(2,1,0)
    seg_B = seg_B[1:len(seg_B)-2, :, :].transpose(2,1,0)
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

import pandas as pd

def get_nice_dataframe(df, metrics, vertical = False):
    models = df["Model"].unique()

    # creates a dataframe where each metric has a column for its mean and std.
    # the mean and std. are computed for each model
    # the dataframe is then saved as a csv file
    df_std = pd.DataFrame(columns=metrics)
    for metric in metrics:
        df_std[metric] = df.groupby(["Model"])[metric].std()

    df_mean = pd.DataFrame(columns=metrics)
    for metric in metrics:
        df_mean[metric] = df.groupby(["Model"])[metric].mean()

    df_mean = df_mean.round(2)
    df_std = df_std.round(2)

    #combine both dataframes, intercalating columns

    empty_df = pd.DataFrame(columns = metrics)
    for metric in metrics:
        i = 0
        for model in models:
            mean_str = str(df_mean.loc[model, metric]) 
            std_str = str(df_std.loc[model, metric]) 
            if len(mean_str) == 3 or (mean_str[2] == '.' and len(mean_str) == 4):
                mean_str += '0'
            if len(std_str) == 3 or (std_str[2] == '.' and len(std_str) == 4):
                std_str += '0'
              
            empty_df.loc[model, metric] = mean_str + " (" + std_str + ")"
            i+=1

    # transposes the dataframe
    if vertical:
        empty_df = empty_df.T
    
    return empty_df