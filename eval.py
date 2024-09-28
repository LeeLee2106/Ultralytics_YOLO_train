import os
import numpy as np
from collections import Counter
import cv2 as cv


def convertToAbsoluteValues(size, box):
    """
    YOLO 형식의 좌표를 VOC 형식 좌표로 변환 (centerX, centerY, width, height -> x_min, y_min, x_max, y_max)
    """
    xIn = round(((2 * float(box[0]) - float(box[2])) * size[0] / 2))
    yIn = round(((2 * float(box[1]) - float(box[3])) * size[1] / 2))
    xEnd = xIn + round(float(box[2]) * size[0])
    yEnd = yIn + round(float(box[3]) * size[1])

    if xIn < 0: xIn = 0
    if yIn < 0: yIn = 0
    if xEnd >= size[0]: xEnd = size[0] - 1
    if yEnd >= size[1]: yEnd = size[1] - 1

    return (xIn, yIn, xEnd, yEnd)



def read_boxes_from_file(filepath, image_width, image_height):
    """
    주어진 파일 경로에서 YOLO 형식 [class, xn, yn, wn, hn] 값을 읽어 절대 좌표로 변환
    """
    print(f"Reading boxes from {filepath}")
    boxes = []
    with open(filepath, 'r') as f:
        for line in f:
            class_id, xn, yn, wn, hn = map(float, line.strip().split())
            x1, y1, x2, y2 = convertToAbsoluteValues((image_width, image_height), (xn, yn, wn, hn))
            boxes.append([class_id, x1, y1, x2, y2])
    print(f"Read {len(boxes)} boxes from {filepath}")
    return boxes

def process_folders(gt_folder, pred_folder, image_width, image_height):
    """
    Ground truth와 예측된 box 정보가 있는 폴더에서 각각 정보를 읽어 AP 및 mAP를 계산.
    """
    gt_files = sorted(os.listdir(gt_folder))
    pred_files = sorted(os.listdir(pred_folder))

    detections, groundtruths = [], []
    classes = set()

    # Ground truth와 예측된 box 정보 읽기
    for gt_file, pred_file in zip(gt_files, pred_files):
        print(f"Processing {gt_file} and {pred_file}")
        gt_filepath = os.path.join(gt_folder, gt_file)
        pred_filepath = os.path.join(pred_folder, pred_file)

        # GT 및 Pred box 읽기
        gt_boxes = read_boxes_from_file(gt_filepath, image_width, image_height)
        pred_boxes = read_boxes_from_file(pred_filepath, image_width, image_height)

        for box in gt_boxes:
            # 각 box를 (x_min, y_min, x_max, y_max)로 저장
            groundtruths.append([gt_file, box[0], 1.0, (box[1], box[2], box[3], box[4])])  # 1.0은 confidence (Ground Truth에서 항상 1)
            classes.add(box[0])  # class id 추가

        for box in pred_boxes:
            # 각 box를 (x_min, y_min, x_max, y_max)로 저장
            detections.append([pred_file, box[0], 1.0, (box[1], box[2], box[3], box[4])])  # Confidence를 1.0으로 설정
            classes.add(box[0])  # class id 추가

    classes = sorted(classes)
    print(f"Total {len(detections)} detections and {len(groundtruths)} ground truths processed.")
    return detections, groundtruths, classes


def getArea(box):
    """Bounding box의 넓이를 계산"""
    return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)

def getIntersectionArea(boxA, boxB):
    """두 box의 겹치는 영역을 계산"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    return max(0, xB - xA + 1) * max(0, yB - yA + 1)

def getUnionAreas(boxA, boxB, interArea=None):
    """두 box의 넓이와 겹치는 영역을 고려하여 합집합 영역 계산"""
    area_A = getArea(boxA)
    area_B = getArea(boxB)
    if interArea is None:
        interArea = getIntersectionArea(boxA, boxB)
    return float(area_A + area_B - interArea)

def boxesIntersect(boxA, boxB):
    """두 box가 겹치는지 여부를 반환"""
    if boxA[0] > boxB[2]: return False  # boxA가 boxB 오른쪽에 위치
    if boxB[0] > boxA[2]: return False  # boxA가 boxB 왼쪽에 위치
    if boxA[3] < boxB[1]: return False  # boxA가 boxB 위쪽에 위치
    if boxA[1] > boxB[3]: return False  # boxA가 boxB 아래쪽에 위치
    return True

def iou(boxA, boxB):
    """두 box의 IoU 계산"""
    if not boxesIntersect(boxA, boxB):
        return 0
    interArea = getIntersectionArea(boxA, boxB)
    union = getUnionAreas(boxA, boxB, interArea=interArea)
    return interArea / union


def AP(detections, groundtruths, classes, IOUThreshold=0.3, method='AP'):
    """
    각 클래스에 대한 AP(Average Precision)를 계산
    """
    result = []
    for c in classes:
        dects = [d for d in detections if d[1] == c]
        gts = [g for g in groundtruths if g[1] == c]
        npos = len(gts)
        dects = sorted(dects, key=lambda conf: conf[2], reverse=True)

        TP = np.zeros(len(dects))
        FP = np.zeros(len(dects))

        det = Counter(cc[0] for cc in gts)
        for key, val in det.items():
            det[key] = np.zeros(val)

        for d in range(len(dects)):
            gt = [gt for gt in gts if gt[0] == dects[d][0]]
            iouMax = 0
            for j in range(len(gt)):
                iou1 = iou(dects[d][3], gt[j][3])
                if iou1 > iouMax:
                    iouMax = iou1
                    jmax = j

            if iouMax >= IOUThreshold:
                if det[dects[d][0]][jmax] == 0:
                    TP[d] = 1
                    det[dects[d][0]][jmax] = 1
                else:
                    FP[d] = 1
            else:
                FP[d] = 1

        acc_FP = np.cumsum(FP)
        acc_TP = np.cumsum(TP)
        rec = acc_TP / npos
        prec = np.divide(acc_TP, (acc_FP + acc_TP))

        if method == "AP":
            [ap, mpre, mrec, ii] = calculateAveragePrecision(rec, prec)
        else:
            [ap, mpre, mrec, _] = ElevenPointInterpolatedAP(rec, prec)

        r = {
            'class': c,
            'precision': prec,
            'recall': rec,
            'AP': ap,
            'interpolated precision': mpre,
            'interpolated recall': mrec,
            'total positives': npos,
            'total TP': np.sum(TP),
            'total FP': np.sum(FP)
        }
        result.append(r)

    return result

def calculateAveragePrecision(rec, prec):
    """Precision-Recall 곡선의 면적을 계산"""
    mrec = [0] + [e for e in rec] + [1]
    mpre = [0] + [e for e in prec] + [0]

    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    ii = [i for i in range(len(mrec) - 1) if mrec[1:][i] != mrec[:-1][i]]
    ap = sum((mrec[i + 1] - mrec[i]) * mpre[i + 1] for i in ii)
    return [ap, mpre, mrec, ii]

def ElevenPointInterpolatedAP(rec, prec):
    """11-point interpolation을 통해 AP를 계산"""
    recallValues = np.linspace(0, 1, 11)[::-1]
    rhoInterp, recallValid = [], []
    mrec = [e for e in rec]
    mpre = [e for e in prec]

    for r in recallValues:
        argGreaterRecalls = np.argwhere(mrec[:] >= r)
        pmax = 0
        if argGreaterRecalls.size != 0:
            pmax = max(mpre[argGreaterRecalls.min():])
        recallValid.append(r)
        rhoInterp.append(pmax)

    ap = sum(rhoInterp) / 11
    return [ap, rhoInterp, recallValues, None]

def calculate_mAP_50_to_95(detections, groundtruths, classes):
    """
    mAP@50부터 mAP@95까지의 평균 값을 계산
    """
    iou_thresholds = np.arange(0.5, 1.0, 0.05)  # 0.5에서 0.95까지 0.05 간격
    mAP_list = []

    for iou_thresh in iou_thresholds:
        print(f"Calculating mAP at IoU threshold: {iou_thresh:.2f}")
        ap_results = AP(detections, groundtruths, classes, IOUThreshold=iou_thresh)
        mAP_list.append(mAP(ap_results))

    # mAP@50-95는 모든 mAP 값들의 평균
    mean_mAP = np.mean(mAP_list)
    print(f"mAP@50-95: {mean_mAP:.4f}")
    return mean_mAP


def mAP(result):
    """모든 클래스에 대한 mAP(mean Average Precision) 계산"""
    ap = 0
    for r in result:
        ap += r['AP']
    return ap / len(result)
def calculate_class_wise_mAP(detections, groundtruths, classes, IOUThreshold):
    """
    각 클래스에 대한 AP 및 mAP를 계산
    """
    ap_results = AP(detections, groundtruths, classes, IOUThreshold=IOUThreshold)
    class_wise_map = {}
    for class_result in ap_results:
        class_wise_map[class_result['class']] = class_result['AP']
        print(f"Class {class_result['class']} - AP at IoU {IOUThreshold:.2f}: {class_result['AP']:.4f}")
    return ap_results, class_wise_map

def calculate_mAP_50(detections, groundtruths, classes):
    """
    mAP@50 계산
    """
    iou_thresh = 0.5
    print(f"Calculating mAP at IoU threshold: {iou_thresh:.2f}")
    ap_results, class_wise_map_50 = calculate_class_wise_mAP(detections, groundtruths, classes, IOUThreshold=iou_thresh)
    mean_mAP_50 = mAP(ap_results)
    print(f"Overall mAP@50: {mean_mAP_50:.4f}")
    return mean_mAP_50, class_wise_map_50

def calculate_mAP_50_to_95_with_class(detections, groundtruths, classes):
    """
    mAP@50부터 mAP@95까지 각 클래스 별 AP와 평균 mAP 값을 계산
    """
    iou_thresholds = np.arange(0.5, 1.0, 0.05)  # 0.5에서 0.95까지 0.05 간격
    mAP_list = []
    class_wise_mAP_50_95 = {c: [] for c in classes}

    for iou_thresh in iou_thresholds:
        print(f"Calculating mAP at IoU threshold: {iou_thresh:.2f}")
        ap_results, class_wise_map = calculate_class_wise_mAP(detections, groundtruths, classes, IOUThreshold=iou_thresh)
        mAP_list.append(mAP(ap_results))
        
        # Store per-class mAP for this IoU threshold
        for c, ap in class_wise_map.items():
            class_wise_mAP_50_95[c].append(ap)

    # mAP@50-95는 모든 mAP 값들의 평균
    mean_mAP_50_95 = np.mean(mAP_list)
    print(f"Overall mAP@50-95: {mean_mAP_50_95:.4f}")
    
    # Calculate mean AP for each class over all IoU thresholds
    for c in classes:
        class_avg_mAP_50_95 = np.mean(class_wise_mAP_50_95[c])
        print(f"Class {c} - mAP@50-95: {class_avg_mAP_50_95:.4f}")
    
    return mean_mAP_50_95, class_wise_mAP_50_95

# 사용 예시
gt_folder = '/root/workspace/usv/src/icra/01_rgb_set/sampled_labels_9_filt'  # GT 파일이 있는 폴더 경로
#pred_folder = '/root/workspace/usv/src/icra/01_rgb_set/yolo_worldl_01_rgb_left_val_box'  # 예측 파일이 있는 폴더 경로
pred_folder = '/root/workspace/usv/src/icra/01_rgb_set/dino_01_rgb_left_val_box'  # 예측 파일이 있는 폴더 경로


image_width = 2048  # 이미지 너비
image_height = 1080  # 이미지 높이

# GT와 Pred 정보를 불러옴
detections, groundtruths, classes = process_folders(gt_folder, pred_folder, image_width, image_height)

# mAP@50 계산
mean_mAP_50, class_wise_map_50 = calculate_mAP_50(detections, groundtruths, classes)

# mAP@50-95 계산
mean_mAP_50_95, class_wise_mAP_50_95 = calculate_mAP_50_to_95_with_class(detections, groundtruths, classes)
