import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.sam_wrapper.load_model import load_sam_model
from src.utils.image_io import load_image, resize_image, ensure_dir
from src.predictor import predict_mask
from src.utils.visualization import (
    save_mask,
    save_prompt_image,
    save_overlay,
    save_three_panel_figure,
)


WINDOW_NAME = "SAM Interactive Point Segmentation"
RESULT_WINDOW_NAME = "SAM Result"


def load_config(config_path: str) -> dict:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def pil_to_bgr(image):
    rgb = np.array(image)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


def overlay_mask_on_bgr(image_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.45):
    mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise ValueError(f"mask shape가 올바르지 않습니다: {mask.shape}")

    result = image_bgr.copy()

    red_overlay = np.zeros_like(result, dtype=np.uint8)
    red_overlay[:, :, 2] = 255  # BGR에서 red channel

    binary = mask.astype(bool)
    result[binary] = cv2.addWeighted(
        result[binary], 1 - alpha,
        red_overlay[binary], alpha,
        0
    )

    return result


def draw_points_on_bgr(image_bgr: np.ndarray, points, labels):
    canvas = image_bgr.copy()

    for (x, y), label in zip(points, labels):
        x, y = int(x), int(y)

        if label == 1:
            # foreground: 빨간 원
            cv2.circle(canvas, (x, y), 8, (0, 0, 255), 2)
        else:
            # background: 파란 X
            cv2.line(canvas, (x - 8, y - 8), (x + 8, y + 8), (255, 0, 0), 2)
            cv2.line(canvas, (x - 8, y + 8), (x + 8, y - 8), (255, 0, 0), 2)

        text = f"({x},{y})"
        cv2.putText(
            canvas,
            text,
            (x + 8, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            text,
            (x + 8, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    info1 = "Left click: FG   Right click: BG   Enter: Run"
    info2 = "r: Reset   s: Save   q: Quit"
    cv2.putText(canvas, info1, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, info1, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, info2, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, info2, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

    return canvas


class InteractiveSegmenter:
    def __init__(self, config: dict):
        self.config = config

        self.model_dir = config["model"]["local_dir"]
        self.requested_device = config["runtime"]["device"]
        self.image_path = config["input"]["image_path"]
        self.output_dir = config["output"]["dir"]

        resize_cfg = config["input"].get("resize", {})
        self.resize_enabled = resize_cfg.get("enabled", False)
        self.resize_size = resize_cfg.get("size", 1024)

        ensure_dir(self.output_dir)

        print("[1/4] 모델 로드 중...")
        self.processor, self.model, self.device = load_sam_model(
            self.model_dir,
            self.requested_device
        )

        print(f"[2/4] 이미지 로드 중... ({self.image_path})")
        self.image = load_image(self.image_path)

        if self.resize_enabled:
            print(f"[2-1/4] resize 적용 중... (long side -> {self.resize_size})")
            self.image, self.scale = resize_image(self.image, self.resize_size)
        else:
            self.scale = 1.0

        print(f"[3/4] 표시 이미지 준비 중... (device={self.device})")
        self.image_bgr = pil_to_bgr(self.image)

        self.points = []
        self.labels = []
        self.last_result = None

        print("[4/4] 준비 완료")
        print("좌클릭: foreground, 우클릭: background, Enter: 실행, r: 초기화, s: 저장, q: 종료")

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append([x, y])
            self.labels.append(1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.points.append([x, y])
            self.labels.append(0)

    def run_segmentation(self):
        if len(self.points) == 0:
            print("점이 없습니다. 먼저 foreground/background 점을 찍어주세요.")
            return

        print("segmentation 실행 중...")
        result = predict_mask(
            image=self.image,
            processor=self.processor,
            model=self.model,
            device=self.device,
            points=self.points,
            labels=self.labels,
        )
        self.last_result = result

        overlay_bgr = overlay_mask_on_bgr(self.image_bgr, result["mask"], alpha=0.45)
        cv2.imshow(RESULT_WINDOW_NAME, overlay_bgr)

        print(f"완료 - score: {result['score']:.4f}")

    def save_outputs(self):
        if self.last_result is None:
            print("저장할 결과가 없습니다. 먼저 Enter로 segmentation을 실행하세요.")
            return

        image_stem = Path(self.image_path).stem

        mask_path = Path(self.output_dir) / f"{image_stem}_mask.png"
        prompt_path = Path(self.output_dir) / f"{image_stem}_prompt.png"
        overlay_path = Path(self.output_dir) / f"{image_stem}_overlay.png"
        panel_path = Path(self.output_dir) / f"{image_stem}_panel.png"

        save_mask(self.last_result["mask"], str(mask_path))
        save_prompt_image(self.image, self.points, self.labels, str(prompt_path))
        save_overlay(self.image, self.last_result["mask"], str(overlay_path))
        save_three_panel_figure(
            original_image=self.image,
            prompt_image_path=str(prompt_path),
            overlay_image_path=str(overlay_path),
            save_path=str(panel_path),
        )

        print("저장 완료")
        print(f"mask   : {mask_path}")
        print(f"prompt : {prompt_path}")
        print(f"overlay: {overlay_path}")
        print(f"panel  : {panel_path}")

    def reset_points(self):
        self.points = []
        self.labels = []
        self.last_result = None
        if cv2.getWindowProperty(RESULT_WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1:
            cv2.destroyWindow(RESULT_WINDOW_NAME)
        print("점과 결과를 초기화했습니다.")

    def run(self):
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, self.mouse_callback)

        while True:
            display = draw_points_on_bgr(self.image_bgr, self.points, self.labels)
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(20) & 0xFF

            if key == 13:  # Enter
                self.run_segmentation()
            elif key == ord("r"):
                self.reset_points()
            elif key == ord("s"):
                self.save_outputs()
            elif key == ord("q"):
                break

        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Interactive point-based segmentation with SAM")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="YAML 설정 파일 경로"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    app = InteractiveSegmenter(config)
    app.run()


if __name__ == "__main__":
    main()

# run
# python scripts/download_checkpoint.py
# python -m src.main --config configs/default.yaml