

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

from config import get_config
from face_detector import FaceDetector, MultipleFacesError
from recognizer import FaceRecognizer
from utils import LOGGER

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CONFIG = get_config()


def iter_person_dirs(dataset_root: Path):
    for entry in sorted(dataset_root.iterdir()):
        if entry.is_dir():
            yield entry


def enroll(dataset_root: Path, output_dir: Path, min_images: int) -> None:
    if not dataset_root.exists():
        LOGGER.error("Dataset directory '%s' does not exist.", dataset_root)
        sys.exit(1)

    detector = FaceDetector()
    recognizer = FaceRecognizer()

    all_embeddings: list[np.ndarray] = []
    all_labels: list[str] = []
    per_person_counts: dict[str, int] = {}

    person_dirs = list(iter_person_dirs(dataset_root))
    if not person_dirs:
        LOGGER.error(
            "No person sub-directories found under '%s'. Expected dataset/<name>/*.jpg", dataset_root
        )
        sys.exit(1)

    for person_dir in person_dirs:
        person_name = person_dir.name
        image_paths = sorted(
            p for p in person_dir.iterdir() if p.suffix.lower() in VALID_EXTENSIONS
        )

        if not image_paths:
            LOGGER.warning("No images found for '%s' — skipping.", person_name)
            continue

        count = 0
        for img_path in image_paths:
            image = cv2.imread(str(img_path))
            if image is None:
                LOGGER.warning("Could not read '%s' — skipping (invalid image).", img_path)
                continue

            try:
                face = detector.detect(image, enforce_single_face=True)
            except MultipleFacesError:
                LOGGER.warning(
                    "Multiple faces detected in '%s' — skipping (enrollment requires exactly one face).",
                    img_path,
                )
                continue

            if face is None:
                LOGGER.warning("No face detected in '%s' — skipping.", img_path)
                continue

            embedding = recognizer.get_embedding(face.embedding_input)
            all_embeddings.append(embedding)
            all_labels.append(person_name)
            count += 1

        per_person_counts[person_name] = count
        if count < min_images:
            LOGGER.warning(
                "Only %d usable image(s) enrolled for '%s' (minimum recommended: %d).",
                count, person_name, min_images,
            )
        else:
            LOGGER.info("Enrolled %d image(s) for '%s'.", count, person_name)

    if not all_embeddings:
        LOGGER.error("No embeddings were generated. Nothing was enrolled.")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_array = np.stack(all_embeddings).astype(np.float32)

    emb_path = output_dir / "reference_embeddings.npy"
    lbl_path = output_dir / "reference_labels.pkl"

    np.save(emb_path, embeddings_array)
    with open(lbl_path, "wb") as f:
        pickle.dump(all_labels, f)

    LOGGER.info(
        "Enrollment complete: %d embeddings across %d identities written to '%s' and '%s'.",
        len(all_labels), len(per_person_counts), emb_path, lbl_path,
    )
    for name, count in per_person_counts.items():
        LOGGER.info("  - %-20s %d image(s)", name, count)


def main() -> None:
    parser = argparse.ArgumentParser(description="Enroll faces from a dataset directory.")
    parser.add_argument(
        "--images", type=str, required=True,
        help="Path to the dataset root, containing one sub-directory per person.",
    )
    parser.add_argument(
        "--output", type=str, default=str(Path(CONFIG.recognition.embeddings_path).parent),
        help="Directory to write reference_embeddings.npy / reference_labels.pkl into.",
    )
    parser.add_argument(
        "--min-images", type=int, default=3,
        help="Warn if a person ends up with fewer than this many usable images.",
    )
    args = parser.parse_args()

    enroll(Path(args.images), Path(args.output), args.min_images)


if __name__ == "__main__":
    main()
