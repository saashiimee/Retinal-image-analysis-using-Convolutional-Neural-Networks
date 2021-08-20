import os
import cv2
import numpy as np
import sklearn.metrics as sm


def get_confusion_matrix_elements(groundtruth_list, predicted_list):

    tn, fp, fn, tp = sm.confusion_matrix(groundtruth_list, predicted_list, labels=[0, 1]).ravel()
    tn, fp, fn, tp = np.float64(tn), np.float64(fp), np.float64(fn), np.float64(tp)

    return tn, fp, fn, tp


def get_prec_rec_IoU_accuracy(groundtruth_list, predicted_list):

    tn, fp, fn, tp = get_confusion_matrix_elements(groundtruth_list, predicted_list)

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    IoU = tp / (tp + fp + fn)

    return prec, rec, IoU, accuracy


def get_f1_score(groundtruth_list, predicted_list):

    tn, fp, fn, tp = get_confusion_matrix_elements(groundtruth_list, predicted_list)

    f1_score = (2 * tp) / ((2 * tp) + fp + fn)

    return f1_score


def get_validation_metrics(groundtruth, predicted):

    print(np.shape(groundtruth))
    print(groundtruth.shape)
    u, v = np.shape(groundtruth)
    groundtruth_list = np.reshape(groundtruth, (u * v,))
    predicted_list = np.reshape(predicted, (u * v,))
    prec, rec, IoU, acc = get_prec_rec_IoU_accuracy(groundtruth_list, predicted_list)
    f1_score = get_f1_score(groundtruth_list, predicted_list)
    # print("Precision=",prec, "Recall=",rec, "IoU=",IoU, "acc=",acc, "F1=",f1_score)
    return prec, rec, IoU, acc, f1_score


def evaluate_result(gth_path, npyfile, target_size=(512, 512)):
    files = sorted(os.listdir(gth_path))
    print(files)
    prec = 0
    rec = 0
    acc = 0
    IoU = 0
    f1_score = 0
    for i, item in enumerate(npyfile):
        img = item[:, :, 0]
        gth = cv2.imread(os.path.join(gth_path, files[i]), cv2.COLOR_BGR2GRAY)
        gth = cv2.resize(gth, target_size)
        img1 = np.array(((img - np.min(img)) / np.ptp(img)) > 0.1).astype(float)
        gth1 = np.array(((gth - np.min(gth)) / np.ptp(gth)) > 0.1).astype(float)
        p, r, I, a, f = get_validation_metrics(gth1, img1)
        prec = prec + p
        rec = rec + r
        acc = acc + a
        IoU = IoU + I
        f1_score = f1_score + f
    print("Precision=", prec / (i + 1), "Recall=", rec / (i + 1), "IoU=", IoU / (i + 1), "Accuracy=", acc / (i + 1), "F1=",
          f1_score / (i + 1))
