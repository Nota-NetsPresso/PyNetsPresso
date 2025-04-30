import json
import random
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
from tqdm import tqdm

from netspresso.clients.dataforge.schemas.response_body import DatasetVersionResponse
from netspresso.trainer.storage.dataforge import Split, dataforge


class DatasetManager:
    """
    Class for managing dataset operations: downloading, organizing,
    and preparing datasets for training and evaluation.
    """

    def __init__(self, token_handler: Optional[Any] = None) -> None:
        self.token_handler: Optional[Any] = token_handler

    def _check_training_dataset_exists(self, dataset_dir: Path) -> bool:
        """
        Check if training dataset already exists with all required directories and files.

        Args:
            dataset_dir: Path to the dataset directory

        Returns:
            bool: True if the dataset exists and is complete
        """
        return (
            dataset_dir.exists()
            and (dataset_dir / "id_mapping.json").exists()
            and (dataset_dir / "images" / "train").exists()
            and (dataset_dir / "labels" / "train").exists()
            and (dataset_dir / "images" / "valid").exists()
            and (dataset_dir / "labels" / "valid").exists()
        )

    def _use_existing_training_dataset(self, dataset_dir: Path) -> str:
        """
        Use existing training dataset.

        Args:
            dataset_dir: Path to the dataset directory

        Returns:
            str: Path to the dataset directory or empty string on error
        """
        logger.info(f"Dataset already exists at {dataset_dir}, using existing files")

        # Count existing files for logging
        train_images: List[Path] = list((dataset_dir / "images" / "train").glob("*"))
        valid_images: List[Path] = list((dataset_dir / "images" / "valid").glob("*"))
        logger.info(f"Found {len(train_images)} training and {len(valid_images)} validation samples")

        return dataset_dir.as_posix()

    def _check_evaluation_dataset_exists(self, dataset_dir: Path, split: str) -> bool:
        """
        Check if evaluation dataset already exists with all required directories and files.

        Args:
            dataset_dir: Path to the dataset directory
            split: Dataset split (e.g., "test")

        Returns:
            bool: True if the dataset exists and is complete
        """
        return (
            dataset_dir.exists()
            and (dataset_dir / "id_mapping.json").exists()
            and (dataset_dir / "images" / split.lower()).exists()
            and (dataset_dir / "labels" / split.lower()).exists()
        )

    def _use_existing_evaluation_dataset(self, dataset_dir: Path, split: str) -> str:
        """
        Use existing evaluation dataset.

        Args:
            dataset_dir: Path to the dataset directory
            split: Dataset split (e.g., "test")

        Returns:
            str: Path to the dataset directory
        """
        logger.info(f"Evaluation dataset already exists at {dataset_dir}, using existing files")

        # Count existing files for logging
        image_files: List[Path] = list((dataset_dir / "images" / split.lower()).glob("*"))
        logger.info(f"Found {len(image_files)} evaluation samples")

        return dataset_dir.as_posix()

    def _get_dataset_version_with_retry(
        self, dataset_uuid: str, split: str, max_retries: int = 3, retry_delay: int = 5
    ) -> Optional[DatasetVersionResponse]:
        """
        Common function: Get dataset version with retry logic.

        Args:
            dataset_uuid: The UUID of the dataset
            split: Dataset split (e.g., TRAIN, TEST)
            max_retries: Maximum number of retry attempts
            retry_delay: Delay in seconds between retry attempts

        Returns:
            DatasetVersionResponse or None if failed
        """
        dataset_version: Optional[DatasetVersionResponse] = None
        permanent_error: bool = False

        for attempt in range(max_retries):
            try:
                dataset_version = dataforge.get_latest_dataset_version(dataset_uuid=dataset_uuid, split=split)
                if not dataset_version or not dataset_version.data:
                    logger.error(f"Could not get dataset info for UUID: {dataset_uuid}, split: {split}")
                    permanent_error = True
                    break
                # Success, break the retry loop
                break
            except FileNotFoundError as e:
                # Permanent error - don't retry
                logger.error(f"Dataset not found (UUID: {dataset_uuid}, split: {split}): {str(e)}")
                permanent_error = True
                break
            except Exception as e:
                # Potentially temporary error - retry
                current_delay: int = retry_delay * (attempt + 1)
                if attempt < max_retries - 1:
                    logger.warning(f"Error getting dataset version (attempt {attempt+1}/{max_retries}): {str(e)}")
                    logger.info(f"Retrying in {current_delay} seconds...")
                    time.sleep(current_delay)
                else:
                    logger.error(f"Failed to get dataset version after {max_retries} attempts: {str(e)}")

        if permanent_error or dataset_version is None:
            return None
        return dataset_version

    def _download_dataset_with_retry(
        self,
        dataset_version: DatasetVersionResponse,
        temp_dir: Path,
        split: str,
        max_retries: int = 3,
        retry_delay: int = 5,
        verbose: bool = False
    ) -> bool:
        """
        Common function: Download dataset with retry logic.

        Args:
            dataset_version: Dataset version response
            temp_dir: Directory to download files to
            split: Dataset split name for logging
            max_retries: Maximum number of retry attempts
            retry_delay: Delay in seconds between retry attempts
            verbose: Whether to log detailed progress

        Returns:
            bool: True if download was successful
        """
        download_success: bool = False
        permanent_download_error: bool = False

        logger.info(f"Downloading {split} data")

        for attempt in range(max_retries):
            try:
                # Download data for this split
                result: bool = dataforge.download_dataset(
                    dataset_version=dataset_version, output_dir=str(temp_dir), verbose=verbose
                )

                if not result:
                    logger.error(f"Failed to download {split} data")
                    permanent_download_error = True
                    break

                download_success = True
                logger.success(f"Successfully downloaded {split} data")
                break
            except FileNotFoundError as e:
                # Permanent error - don't retry
                logger.error(f"Dataset files not found: {str(e)}")
                permanent_download_error = True
                break
            except Exception as e:
                # Potentially temporary error - retry
                current_delay: int = retry_delay * (attempt + 1)
                if attempt < max_retries - 1:
                    logger.warning(f"Error downloading dataset (attempt {attempt+1}/{max_retries}): {str(e)}")
                    logger.info(f"Retrying in {current_delay} seconds...")
                    time.sleep(current_delay)
                else:
                    logger.error(f"Failed to download dataset after {max_retries} attempts: {str(e)}")

        return not (permanent_download_error or not download_success)

    def _save_id_mapping(self, dataset_version: DatasetVersionResponse, output_path: Path) -> Dict[str, str]:
        """
        Common function: Save ID mapping file.

        Args:
            dataset_version: Dataset version response
            output_path: Path where the id_mapping.json will be saved

        Returns:
            dict: The ID mapping that was saved
        """
        try:
            id_mapping: Dict[str, str] = dataset_version.data.dataset_metadata.id_mapping
            with open(output_path, "w") as f:
                json.dump(id_mapping, f)
            logger.info(f"Saved id_mapping.json with {len(id_mapping)} classes")
            return id_mapping
        except Exception as e:
            logger.warning(f"Error saving id_mapping.json: {str(e)}")
            # Create a default mapping if necessary
            default_mapping: Dict[str, str] = {"0": "background", "1": "object"}
            with open(output_path, "w") as f:
                json.dump(default_mapping, f)
            return default_mapping

    def _cleanup_temp_dir(self, temp_dir: Path) -> None:
        """
        Common function: Clean up temporary directory.

        Args:
            temp_dir: Path to temporary directory
        """
        try:
            logger.info("Cleaning up temporary files")
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as e:
            logger.warning(f"Error cleaning up temporary files: {str(e)}")

    def _copy_file_with_error_handling(self, src_file: Path, dest_file: Path) -> bool:
        """
        Copy a file with error handling.

        Args:
            src_file: Source file path
            dest_file: Destination file path

        Returns:
            bool: True if copy was successful
        """
        try:
            shutil.copy2(src_file, dest_file)
            return True
        except Exception as e:
            logger.warning(f"Error copying file {src_file.name}: {str(e)}")
            return False

    def download_dataset_for_training(
        self,
        dataset_uuid: str,
        output_dir: str = "./datasets",
        valid_split: float = 0.2,
        random_seed: int = 0,
        max_retries: int = 3,
        retry_delay: int = 5,
        verbose: bool = False,
    ) -> str:
        """
        Download dataset from DataForge and prepare it for training.

        Args:
            dataset_uuid: The UUID of the dataset to download
            output_dir: Directory to save downloaded files
            valid_split: Ratio of validation data to split from train data (0.0-1.0)
            random_seed: Random seed for reproducible splitting
            max_retries: Maximum number of retry attempts for network/storage errors
            retry_delay: Delay in seconds between retry attempts (will increase with each retry)
            verbose: Whether to log detailed progress for each file (default: False)

        Returns:
            str: Path to the configured dataset
        """
        try:
            # Create base output directory
            dataset_dir: Path = Path(output_dir) / dataset_uuid

            # Check if dataset already exists
            if self._check_training_dataset_exists(dataset_dir):
                return self._use_existing_training_dataset(dataset_dir)

            # Dataset doesn't exist or is incomplete, proceed with download
            dataset_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Downloading dataset with UUID: {dataset_uuid}")

            # Get the latest dataset version with retry logic
            dataset_version: Optional[DatasetVersionResponse] = self._get_dataset_version_with_retry(
                dataset_uuid, Split.TRAIN, max_retries, retry_delay
            )

            if dataset_version is None:
                return ""

            # Create temporary directory for downloads
            temp_dir: Path = dataset_dir / "temp_download"
            temp_dir.mkdir(parents=True, exist_ok=True)

            # Download data with retry logic
            if not self._download_dataset_with_retry(
                dataset_version, temp_dir, Split.TRAIN, max_retries, retry_delay, verbose
            ):
                return ""

            # Prepare directory structure for trainer
            # The trainer expects:
            # - images/train/ and images/valid/ for images
            # - labels/train/ and labels/valid/ for labels
            images_dir: Path = dataset_dir / "images"
            labels_dir: Path = dataset_dir / "labels"

            # Create train/valid directories
            train_images_dir: Path = images_dir / "train"
            train_labels_dir: Path = labels_dir / "train"
            valid_images_dir: Path = images_dir / "valid"
            valid_labels_dir: Path = labels_dir / "valid"

            for dir_path in [train_images_dir, train_labels_dir, valid_images_dir, valid_labels_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)

            # Save id_mapping
            self._save_id_mapping(dataset_version, dataset_dir / "id_mapping.json")

            # Get source file paths
            source_images_dir: Path = temp_dir / dataset_uuid / "images"
            source_annotations_dir: Path = temp_dir / dataset_uuid / "annotations"

            if not source_images_dir.exists() or not source_annotations_dir.exists():
                logger.error("Required source directories not found after download")
                return ""

            # Get list of all images and corresponding annotations
            image_files: List[Path] = [f for f in source_images_dir.iterdir() if f.is_file()]

            if not image_files:
                logger.error("No image files found in downloaded dataset")
                return ""

            logger.info(f"Found {len(image_files)} image files")

            # Get corresponding annotation files (maintain image-annotation pairing)
            file_pairs: List[Tuple[Path, Path]] = []
            for img_file in image_files:
                # Find matching annotation file (assuming same name, different extension)
                ann_candidates: List[Path] = list(source_annotations_dir.glob(f"{img_file.stem}.*"))
                if ann_candidates:
                    file_pairs.append((img_file, ann_candidates[0]))
                else:
                    logger.warning(f"No matching annotation found for {img_file.name}")

            logger.info(f"Found {len(file_pairs)} valid image-annotation pairs")

            # Randomize and split the dataset
            random.seed(random_seed)
            random.shuffle(file_pairs)

            # Calculate split point
            valid_count: int = max(1, int(len(file_pairs) * valid_split))
            valid_pairs: List[Tuple[Path, Path]] = file_pairs[:valid_count]
            train_pairs: List[Tuple[Path, Path]] = file_pairs[valid_count:]

            logger.info(f"Splitting into {len(train_pairs)} training and {len(valid_pairs)} validation samples")

            # Initialize progress tracking for copying files
            total_pairs: int = len(train_pairs) + len(valid_pairs)

            progress_bar = tqdm(total=total_pairs, desc="Copying files", unit="files")

            # Set logging interval for large datasets
            log_interval: int = max(1, min(1000, total_pairs // 10))  # Log at most 10 times

            # Copy files with better error handling
            copy_success_count: int = 0
            copy_error_count: int = 0
            processed_count: int = 0

            # Copy training files
            for img_file, ann_file in train_pairs:
                img_success: bool = self._copy_file_with_error_handling(img_file, train_images_dir / img_file.name)
                ann_success: bool = self._copy_file_with_error_handling(ann_file, train_labels_dir / ann_file.name)

                if img_success and ann_success:
                    copy_success_count += 1
                    if verbose:
                        logger.debug(f"Copied training file: {img_file.name}")
                else:
                    copy_error_count += 1

                processed_count += 1
                # Update progress
                progress_bar.update(1)
                if processed_count % log_interval == 0 or processed_count == total_pairs:
                    progress_pct: float = 100 * processed_count / total_pairs
                    logger.info(f"Progress: {processed_count}/{total_pairs} files processed ({progress_pct:.1f}%)")

            # Copy validation files
            for img_file, ann_file in valid_pairs:
                img_success: bool = self._copy_file_with_error_handling(img_file, valid_images_dir / img_file.name)
                ann_success: bool = self._copy_file_with_error_handling(ann_file, valid_labels_dir / ann_file.name)

                if img_success and ann_success:
                    copy_success_count += 1
                    if verbose:
                        logger.debug(f"Copied validation file: {img_file.name}")
                else:
                    copy_error_count += 1

                processed_count += 1
                # Update progress
                progress_bar.update(1)
                if processed_count % log_interval == 0 or processed_count == total_pairs:
                    progress_pct: float = 100 * processed_count / total_pairs
                    logger.info(f"Progress: {processed_count}/{total_pairs} files processed ({progress_pct:.1f}%)")

            # Close progress bar if used
            if progress_bar:
                progress_bar.close()

            if copy_error_count > 0:
                logger.warning(
                    f"Encountered {copy_error_count} errors while copying files (successfully copied {copy_success_count} files)"
                )

            # Clean up temporary files
            self._cleanup_temp_dir(temp_dir)

            logger.success(f"Dataset downloaded, split and configured at: {dataset_dir}")
            logger.info(f"Train samples: {len(train_pairs)}, Validation samples: {len(valid_pairs)}")
            return dataset_dir.as_posix()

        except Exception as e:
            logger.exception(f"Unexpected error in download_dataset_for_training: {str(e)}")
            return ""

    def download_dataset_for_evaluation(
        self,
        dataset_uuid: str,
        output_dir: str = "./datasets",
        split: str = Split.TEST,
        max_retries: int = 3,
        retry_delay: int = 5,
        verbose: bool = False,
    ) -> str:
        """
        Download dataset from DataForge for evaluation purposes

        Args:
            dataset_uuid: The UUID of the dataset to download
            output_dir: Directory to save downloaded files
            split: Dataset split to download (default: TEST)
            max_retries: Maximum number of retry attempts for network/storage errors
            retry_delay: Delay in seconds between retry attempts (will increase with each retry)
            verbose: Whether to log detailed progress for each file (default: False)

        Returns:
            str: Path to the configured evaluation dataset
        """
        try:
            # Create base output directory
            dataset_dir: Path = Path(output_dir) / f"{dataset_uuid}_{split}"

            # Check if dataset already exists
            if self._check_evaluation_dataset_exists(dataset_dir, split):
                return self._use_existing_evaluation_dataset(dataset_dir, split)

            # Dataset doesn't exist or is incomplete, proceed with download
            dataset_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Downloading evaluation dataset with UUID: {dataset_uuid}, split: {split}")

            # Get the dataset version with retry logic
            dataset_version: Optional[DatasetVersionResponse] = self._get_dataset_version_with_retry(
                dataset_uuid, split, max_retries, retry_delay
            )

            if dataset_version is None:
                return ""

            # Create temporary directory for downloads
            temp_dir: Path = dataset_dir / "temp_download"
            temp_dir.mkdir(parents=True, exist_ok=True)

            # Download evaluation data with retry logic
            if not self._download_dataset_with_retry(
                dataset_version, temp_dir, split, max_retries, retry_delay, verbose
            ):
                return ""

            # Prepare directory structure for evaluation with test subdirectory
            # Structure with:
            # - images/test/ for test images
            # - labels/test/ for test labels
            images_dir: Path = dataset_dir / "images"
            labels_dir: Path = dataset_dir / "labels"

            # Create test subdirectories
            test_images_dir: Path = images_dir / split.lower()
            test_labels_dir: Path = labels_dir / split.lower()

            # Create directories
            test_images_dir.mkdir(parents=True, exist_ok=True)
            test_labels_dir.mkdir(parents=True, exist_ok=True)

            # Save id_mapping
            _: Dict[str, str] = self._save_id_mapping(dataset_version, dataset_dir / "id_mapping.json")

            # Get source file paths
            source_images_dir: Path = temp_dir / dataset_uuid / "images"
            source_annotations_dir: Path = temp_dir / dataset_uuid / "annotations"

            if not source_images_dir.exists() or not source_annotations_dir.exists():
                logger.error("Required source directories not found after download")
                return ""

            # Get list of all images and corresponding annotations
            image_files: List[Path] = [f for f in source_images_dir.iterdir() if f.is_file()]

            if not image_files:
                logger.error("No image files found in downloaded dataset")
                return ""

            logger.info(f"Found {len(image_files)} image files")

            # Get corresponding annotation files
            file_pairs: List[Tuple[Path, Path]] = []
            for img_file in image_files:
                # Find matching annotation file (assuming same name, different extension)
                ann_candidates: List[Path] = list(source_annotations_dir.glob(f"{img_file.stem}.*"))
                if ann_candidates:
                    file_pairs.append((img_file, ann_candidates[0]))
                else:
                    logger.warning(f"No matching annotation found for {img_file.name}")

            logger.info(f"Found {len(file_pairs)} valid image-annotation pairs")

            # Copy files with error handling and progress tracking
            copy_success_count: int = 0
            copy_error_count: int = 0
            skipped_count: int = 0
            total_files: int = len(file_pairs)

            # Initialize progress tracking
            progress_bar = tqdm(total=total_files, desc="Copying files", unit="files")

            # Set logging interval for large datasets (report every X files)
            log_interval: int = max(1, min(1000, total_files // 10))  # Log at most 10 times for the entire process

            # Copy all files to test subdirectories
            for i, (img_file, ann_file) in enumerate(file_pairs):
                try:
                    # Check if files already exist at destination
                    dest_img_file: Path = test_images_dir / img_file.name
                    dest_ann_file: Path = test_labels_dir / ann_file.name

                    if dest_img_file.exists() and dest_ann_file.exists():
                        skipped_count += 1
                        if verbose:
                            logger.debug(f"Skipping existing file pair: {img_file.name}")
                    else:
                        img_success: bool = self._copy_file_with_error_handling(img_file, dest_img_file)
                        ann_success: bool = self._copy_file_with_error_handling(ann_file, dest_ann_file)

                        if img_success and ann_success:
                            copy_success_count += 1
                            if verbose:
                                logger.debug(f"Copied file pair: {img_file.name}")
                        else:
                            copy_error_count += 1

                except Exception as e:
                    copy_error_count += 1
                    logger.warning(f"Error processing file pair {img_file.name}: {str(e)}")

                # Update progress
                progress_bar.update(1)
                if (i + 1) % log_interval == 0 or (i + 1) == total_files:
                    progress_pct: float = 100 * (i + 1) / total_files
                    logger.info(f"Progress: {i + 1}/{total_files} files processed ({progress_pct:.1f}%)")

            # Close progress bar if used
            if progress_bar:
                progress_bar.close()

            # Log summary statistics
            if copy_error_count > 0:
                logger.warning(f"Encountered {copy_error_count} errors while copying files")
            if skipped_count > 0:
                logger.info(f"Skipped {skipped_count} already existing files")

            logger.success(f"Successfully copied {copy_success_count} files")

            # Clean up temporary files
            self._cleanup_temp_dir(temp_dir)

            logger.success(f"Evaluation dataset downloaded at: {dataset_dir}")
            logger.info(f"Total evaluation samples: {copy_success_count + skipped_count}")

            return dataset_dir.as_posix()

        except Exception as e:
            logger.exception(f"Unexpected error in download_dataset_for_evaluation: {str(e)}")
            return ""
