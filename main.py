import argparse
import copy
import os
import platform
import random
import warnings
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm.auto import tqdm


CLASS_NAMES = {
    0: "Lettuce",
    1: "Potato",
    2: "Carrot",
    3: "Onion",
    4: "Garlic",
    5: "Leek",
    6: "Broccoli",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train a vegetable object detector.")
    parser.add_argument("--dataset-zip", type=str, default=os.environ.get("DATASET_ZIP"), help="Path to dataset.zip")
    parser.add_argument("--data-root", type=str, default=os.environ.get("DATA_ROOT"), help="Path to an extracted dataset root")
    parser.add_argument("--extract-dir", type=str, default=os.environ.get("EXTRACT_DIR", "/tmp/vegetable-dataset"), help="Directory used when unzipping dataset.zip")
    parser.add_argument("--output-dir", type=str, default=os.environ.get("OUTPUT_DIR", "/outputs"), help="Directory for checkpoints and artifacts")
    parser.add_argument("--dataset-percent", type=float, default=float(os.environ.get("DATASET_PERCENT", 100.0)), help="Percent of the dataset to use")
    parser.add_argument("--val-percent", type=float, default=float(os.environ.get("VAL_PERCENT", 20.0)), help="Validation split percent")
    parser.add_argument("--test-percent", type=float, default=float(os.environ.get("TEST_PERCENT", 20.0)), help="Test split percent")
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", 4)), help="Batch size")
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", 3)), help="Training epochs")
    parser.add_argument("--lr", type=float, default=float(os.environ.get("LR", 0.005)), help="Learning rate")
    parser.add_argument("--momentum", type=float, default=float(os.environ.get("MOMENTUM", 0.9)), help="SGD momentum")
    parser.add_argument("--weight-decay", type=float, default=float(os.environ.get("WEIGHT_DECAY", 0.0005)), help="Weight decay")
    parser.add_argument("--lr-step-size", type=int, default=int(os.environ.get("LR_STEP_SIZE", 3)), help="StepLR step size")
    parser.add_argument("--lr-gamma", type=float, default=float(os.environ.get("LR_GAMMA", 0.1)), help="StepLR gamma")
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", 42)), help="Random seed")
    parser.add_argument("--preload", action=argparse.BooleanOptionalAction, default=os.environ.get("PRELOAD", "1") not in {"0", "false", "False"}, help="Preload images into RAM")
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("NUM_WORKERS", 0)), help="DataLoader workers")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=os.environ.get("PRETRAINED", "0") in {"1", "true", "True"}, help="Use pretrained torchvision weights")
    parser.add_argument("--download-if-missing", action=argparse.BooleanOptionalAction, default=os.environ.get("DOWNLOAD_IF_MISSING", "0") in {"1", "true", "True"}, help="Try kagglehub if no local dataset is present")
    parser.add_argument("--run-eval", action=argparse.BooleanOptionalAction, default=os.environ.get("RUN_EVAL", "1") not in {"0", "false", "False"}, help="Run mAP evaluation after training")
    parser.add_argument("--iou-threshold", type=float, default=float(os.environ.get("IOU_THRESHOLD", 0.5)), help="IoU threshold for mAP")
    parser.add_argument("--score-threshold", type=float, default=float(os.environ.get("SCORE_THRESHOLD", 0.05)), help="Score threshold for mAP")
    parser.add_argument("--save-name", type=str, default=os.environ.get("SAVE_NAME", "best_model.pth"), help="Checkpoint filename")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def unzip_dataset(zip_path, extract_dir):
    zip_path = Path(zip_path)
    extract_dir = Path(extract_dir)
    ensure_dir(extract_dir)
    marker = extract_dir / ".extracted"
    if marker.exists():
        return extract_dir
    print(f"Extracting {zip_path} to {extract_dir}...")
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(extract_dir)
    marker.write_text("ok", encoding="utf-8")
    return extract_dir


def find_dataset_root(root_candidate):
    root_candidate = Path(root_candidate)
    for candidate in [root_candidate, root_candidate / "dataset"]:
        if (candidate / "images").is_dir() and (candidate / "labels").is_dir():
            return candidate

    for candidate in root_candidate.rglob("*"):
        if candidate.is_dir() and (candidate / "images").is_dir() and (candidate / "labels").is_dir():
            return candidate

    raise FileNotFoundError(
        f"Could not find images/ and labels/ folders under {root_candidate}. "
        "Expected a dataset root or a dataset/ subfolder."
    )


def resolve_dataset_root(args):
    if args.data_root:
        return find_dataset_root(args.data_root)

    if args.dataset_zip:
        extracted_root = unzip_dataset(args.dataset_zip, args.extract_dir)
        return find_dataset_root(extracted_root)

    if args.download_if_missing:
        try:
            import kagglehub

            path = kagglehub.dataset_download("ayyuce/vegetables")
            return find_dataset_root(path)
        except Exception as exc:
            warnings.warn(f"Dataset download failed: {exc}")

    raise ValueError("Provide --dataset-zip, --data-root, or --download-if-missing.")


def build_image_list(images_dir, dataset_percent, seed):
    image_files = [name for name in os.listdir(images_dir) if name.lower().endswith((".png", ".jpg", ".jpeg"))]
    if not image_files:
        raise FileNotFoundError(f"No images found in {images_dir}")

    rng = random.Random(seed)
    rng.shuffle(image_files)

    if dataset_percent <= 0 or dataset_percent > 100:
        raise ValueError("--dataset-percent must be in the range (0, 100].")

    keep_count = max(1, int(len(image_files) * (dataset_percent / 100.0)))
    return image_files[:keep_count]


class VegetableDataset(Dataset):
    def __init__(self, image_files, images_dir, labels_dir, image_transforms=None):
        self.image_files = list(image_files)
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_transforms = image_transforms

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        img = Image.open(img_path).convert("RGB")

        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(self.labels_dir, label_name)

        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    data = list(map(float, line.split()))
                    class_id = int(data[0])
                    x_c, y_c, box_w, box_h = data[1:]
                    width, height = img.size
                    xmin = (x_c - box_w / 2) * width
                    ymin = (y_c - box_h / 2) * height
                    xmax = (x_c + box_w / 2) * width
                    ymax = (y_c + box_h / 2) * height
                    if xmax > xmin and ymax > ymin:
                        boxes.append([xmin, ymin, xmax, ymax])
                        labels.append(class_id + 1)

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }

        if self.image_transforms:
            img = self.image_transforms(img)

        return img, target

    def __len__(self):
        return len(self.image_files)


class PreloadedVegetableDataset(Dataset):
    def __init__(self, image_files, images_dir, labels_dir, image_transforms=None):
        self.image_files = list(image_files)
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_transforms = image_transforms
        self.images = []
        self.targets = []

        print(f"Preloading {len(self.image_files)} images into RAM...")
        for idx, img_name in enumerate(tqdm(self.image_files, desc="Preloading")):
            img_path = self.images_dir / img_name
            img = Image.open(img_path).convert("RGB")
            label_name = os.path.splitext(img_name)[0] + ".txt"
            label_path = self.labels_dir / label_name

            boxes = []
            labels = []

            if label_path.exists():
                with label_path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        data = list(map(float, line.split()))
                        class_id = int(data[0])
                        x_c, y_c, box_w, box_h = data[1:]
                        width, height = img.size
                        xmin = (x_c - box_w / 2) * width
                        ymin = (y_c - box_h / 2) * height
                        xmax = (x_c + box_w / 2) * width
                        ymax = (y_c + box_h / 2) * height
                        if xmax > xmin and ymax > ymin:
                            boxes.append([xmin, ymin, xmax, ymax])
                            labels.append(class_id + 1)

            self.images.append(img)
            self.targets.append(
                {
                    "boxes": torch.as_tensor(boxes, dtype=torch.float32),
                    "labels": torch.as_tensor(labels, dtype=torch.int64),
                    "image_id": torch.tensor([idx]),
                }
            )

    def __getitem__(self, idx):
        img = self.images[idx]
        target = self.targets[idx]
        if self.image_transforms:
            img = self.image_transforms(img)
        return img, target

    def __len__(self):
        return len(self.images)


def collate_fn(batch):
    return tuple(zip(*batch))


def split_files(image_files, val_percent, test_percent, seed):
    if val_percent < 0 or test_percent < 0 or val_percent + test_percent >= 100:
        raise ValueError("--val-percent and --test-percent must be non-negative and sum to less than 100.")

    if test_percent == 0:
        train_val_files, test_files = list(image_files), []
    else:
        test_fraction = test_percent / 100.0
        train_val_files, test_files = train_test_split(image_files, test_size=test_fraction, random_state=seed)

    if val_percent == 0:
        return train_val_files, [], test_files

    remaining_fraction = 1.0 if test_percent == 0 else 1.0 - (test_percent / 100.0)
    val_fraction = (val_percent / 100.0) / remaining_fraction
    train_files, val_files = train_test_split(train_val_files, test_size=val_fraction, random_state=seed)
    return train_files, val_files, test_files


def make_dataset_class(preload):
    return PreloadedVegetableDataset if preload else VegetableDataset


def make_loader(dataset, batch_size, shuffle, num_workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )


def get_model(num_model_classes, pretrained=False):
    if pretrained:
        weights = models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    else:
        model = models.detection.fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_model_classes)
    return model


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    total_loss = 0.0
    pbar = tqdm(data_loader, total=len(data_loader), desc=f"Epoch {epoch + 1} [Train]")

    for images, targets in pbar:
        images = [image.to(device) for image in images]
        targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
        pbar.set_postfix(loss=f"{losses.item():.4f}")

    return total_loss / len(data_loader) if len(data_loader) > 0 else float("inf")


@torch.no_grad()
def validate_one_epoch(model, data_loader, device):
    was_training = model.training
    model.train()
    total_loss = 0.0
    pbar = tqdm(data_loader, total=len(data_loader), desc="Validating")

    for images, targets in pbar:
        images = [image.to(device) for image in images]
        targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()
        pbar.set_postfix(loss=f"{losses.item():.4f}")

    if not was_training:
        model.eval()

    return total_loss / len(data_loader) if len(data_loader) > 0 else float("inf")


def box_iou(box_a, box_b):
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def compute_ap(rec, prec):
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for idx in range(mpre.size - 2, -1, -1):
        mpre[idx] = max(mpre[idx], mpre[idx + 1])
    change_points = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[change_points + 1] - mrec[change_points]) * mpre[change_points + 1])


def evaluate_map(model, dataloader, device, num_model_classes, iou_threshold=0.5, score_threshold=0.05):
    model.eval()
    cls_scores = {cls_id: [] for cls_id in range(1, num_model_classes)}
    cls_matches = {cls_id: [] for cls_id in range(1, num_model_classes)}
    cls_num_gts = {cls_id: 0 for cls_id in range(1, num_model_classes)}

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating"):
            images = [image.to(device) for image in images]
            outputs = model(images)

            for index, output in enumerate(outputs):
                pred_boxes = output.get("boxes", torch.empty((0, 4))).cpu().numpy()
                pred_labels = output.get("labels", torch.empty((0,), dtype=torch.int64)).cpu().numpy()
                pred_scores = output.get("scores", torch.empty((0,))).cpu().numpy()
                gt_boxes = targets[index]["boxes"].cpu().numpy()
                gt_labels = targets[index]["labels"].cpu().numpy()

                for cls_id in range(1, num_model_classes):
                    gt_mask = gt_labels == cls_id
                    gt_boxes_cls = gt_boxes[gt_mask]
                    cls_num_gts[cls_id] += len(gt_boxes_cls)

                    pred_mask = pred_labels == cls_id
                    boxes_cls = pred_boxes[pred_mask]
                    scores_cls = pred_scores[pred_mask]
                    order = np.argsort(-scores_cls)
                    boxes_cls = boxes_cls[order]
                    scores_cls = scores_cls[order]

                    matched = np.zeros(len(gt_boxes_cls), dtype=bool)
                    for pred_box, score in zip(boxes_cls, scores_cls):
                        if score < score_threshold:
                            continue

                        if len(gt_boxes_cls) == 0:
                            cls_scores[cls_id].append(score)
                            cls_matches[cls_id].append(0)
                            continue

                        ious = np.array([box_iou(pred_box, gt_box) for gt_box in gt_boxes_cls])
                        match_index = ious.argmax()
                        if ious[match_index] >= iou_threshold and not matched[match_index]:
                            cls_scores[cls_id].append(score)
                            cls_matches[cls_id].append(1)
                            matched[match_index] = True
                        else:
                            cls_scores[cls_id].append(score)
                            cls_matches[cls_id].append(0)

    ap_per_class = {}
    for cls_id in range(1, num_model_classes):
        scores = np.array(cls_scores[cls_id])
        matches = np.array(cls_matches[cls_id])
        npos = cls_num_gts[cls_id]

        if npos == 0:
            ap_per_class[cls_id] = None
            continue
        if scores.size == 0:
            ap_per_class[cls_id] = 0.0
            continue

        order = np.argsort(-scores)
        matches = matches[order]
        tp = np.cumsum(matches == 1)
        fp = np.cumsum(matches == 0)
        rec = tp / npos
        prec = tp / (tp + fp + 1e-9)
        ap_per_class[cls_id] = compute_ap(rec, prec)

    valid_aps = [ap for ap in ap_per_class.values() if ap is not None]
    mean_ap = float(np.mean(valid_aps)) if valid_aps else 0.0
    return mean_ap, ap_per_class


def build_dataloaders(dataset_root, args):
    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels"
    image_files = build_image_list(images_dir, args.dataset_percent, args.seed)
    train_files, val_files, test_files = split_files(image_files, args.val_percent, args.test_percent, args.seed)

    data_transforms = transforms.Compose([transforms.ToTensor()])
    dataset_class = make_dataset_class(args.preload)
    train_dataset = dataset_class(train_files, images_dir, labels_dir, data_transforms)
    val_dataset = dataset_class(val_files, images_dir, labels_dir, data_transforms)
    test_dataset = VegetableDataset(test_files, images_dir, labels_dir, data_transforms)

    train_loader = make_loader(train_dataset, args.batch_size, True, args.num_workers)
    val_loader = make_loader(val_dataset, args.batch_size, False, args.num_workers)
    test_loader = make_loader(test_dataset, max(1, min(3, args.batch_size)), False, args.num_workers)
    return train_loader, val_loader, test_loader, len(image_files)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = resolve_device()
    print(f"Using device: {device}")
    print(f"Host platform: {platform.system()}")

    dataset_root = resolve_dataset_root(args)
    print(f"Dataset root: {dataset_root}")

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    train_loader, val_loader, test_loader, total_images = build_dataloaders(dataset_root, args)
    print(f"Loaded {total_images} images after dataset filtering.")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")

    num_model_classes = len(CLASS_NAMES) + 1
    model = get_model(num_model_classes, pretrained=args.pretrained)
    model.to(device)

    optimizer = torch.optim.SGD(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    checkpoint_path = output_dir / args.save_name

    print(f"Starting training for {args.epochs} epoch(s)...")
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        val_loss = validate_one_epoch(model, val_loader, device)
        print(f"Epoch {epoch + 1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        scheduler.step()

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, checkpoint_path)
            print(f"Saved best checkpoint to {checkpoint_path}")

        print("-" * 40)

    model.load_state_dict(best_model_wts)
    print(f"Training complete. Best validation loss: {best_loss:.4f}")

    if args.run_eval:
        mean_ap, ap_per_class = evaluate_map(
            model,
            val_loader,
            device,
            num_model_classes,
            iou_threshold=args.iou_threshold,
            score_threshold=args.score_threshold,
        )
        print(f"mAP@0.5: {mean_ap:.4f}")
        for cls_id, ap in ap_per_class.items():
            class_name = CLASS_NAMES.get(cls_id - 1, f"ID: {cls_id - 1}")
            if ap is None:
                print(f"Class {cls_id} ({class_name}): AP=N/A")
            else:
                print(f"Class {cls_id} ({class_name}): AP={ap:.4f}")


if __name__ == "__main__":
    main()
