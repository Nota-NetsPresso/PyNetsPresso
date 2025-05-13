import json
import random
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger
from tqdm import tqdm

from netspresso.clients.auth.client import TokenHandler
from netspresso.clients.dataforge.schemas.response_body import (
    DatasetPayload,
    DatasetVersionInfo,
    DatasetVersionResponse,
)
from netspresso.exceptions.dataset import DatasetDownloadError, DatasetNotFoundError, DatasetPrepareError
from netspresso.trainer.storage.dataforge import Split, dataforge


class DatasetManager:
    """
    Class for managing dataset operations: downloading, organizing,
    and preparing datasets for training and evaluation.
    """

    def __init__(self, token_handler: TokenHandler) -> None:
        self.token_handler: TokenHandler = token_handler

    def _check_dataset_exists(self, dataset_dir: Path, split: str) -> bool:
        """
        Check if dataset already exists with all required directories and files.

        Args:
            dataset_dir: Path to the dataset directory
            split: Dataset split to check (e.g., "TRAIN", "TEST")

        Returns:
            bool: True if the dataset exists and is complete
        """
        is_for_training = (split == Split.TRAIN)

        if is_for_training:
            # For training, we need both train and valid directories
            return (
                dataset_dir.exists()
                and (dataset_dir / "id_mapping.json").exists()
                and (dataset_dir / "images" / "train").exists()
                and (dataset_dir / "labels" / "train").exists()
                and (dataset_dir / "images" / "valid").exists()
                and (dataset_dir / "labels" / "valid").exists()
            )
        else:
            # For evaluation, we need the specific split directory
            split_dir = "test" if split == Split.TEST else split.lower()
            return (
                dataset_dir.exists()
                and (dataset_dir / "id_mapping.json").exists()
                and (dataset_dir / "images" / split_dir).exists()
                and (dataset_dir / "labels" / split_dir).exists()
            )

    def _use_existing_dataset(self, dataset_dir: Path, split: str) -> str:
        """
        Use existing dataset.

        Args:
            dataset_dir: Path to the dataset directory
            split: Dataset split (e.g., "TRAIN", "TEST")

        Returns:
            str: Path to the dataset directory
        """
        is_for_training = (split == Split.TRAIN)

        if is_for_training:
            logger.info(f"Dataset already exists at {dataset_dir}, using existing files")
            # Count existing files for logging
            train_images: List[Path] = list((dataset_dir / "images" / "train").glob("*"))
            valid_images: List[Path] = list((dataset_dir / "images" / "valid").glob("*"))
            logger.info(f"Found {len(train_images)} training and {len(valid_images)} validation samples")
        else:
            logger.info(f"Evaluation dataset already exists at {dataset_dir}, using existing files")
            split_dir = "test" if split == Split.TEST else split.lower()
            # Count existing files for logging
            image_files: List[Path] = list((dataset_dir / "images" / split_dir).glob("*"))
            logger.info(f"Found {len(image_files)} evaluation samples")

        return dataset_dir.as_posix()

    def _find_file_pairs(self, source_images_dir: Path, source_annotations_dir: Path) -> List[Tuple[Path, Path]]:
        """
        Find matching image and annotation file pairs.

        Args:
            source_images_dir: Directory containing image files
            source_annotations_dir: Directory containing annotation files

        Returns:
            List[Tuple[Path, Path]]: List of (image_file, annotation_file) pairs
        """
        # Get list of all images
        image_files: List[Path] = [f for f in source_images_dir.iterdir() if f.is_file()]

        if not image_files:
            logger.error("No image files found in downloaded dataset")
            return []

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
        return file_pairs

    def _create_split_directories(self, base_dir: Path, split: str) -> Dict[str, Tuple[Path, Path]]:
        """
        Create appropriate directory structure for the dataset.

        Args:
            base_dir: Base directory for creating the structure
            split: Dataset split ("TRAIN" or other)

        Returns:
            Dict: Dictionary mapping split names to (images_dir, labels_dir) tuples
        """
        is_for_training = (split == Split.TRAIN)
        images_dir: Path = base_dir / "images"
        labels_dir: Path = base_dir / "labels"
        result = {}

        if is_for_training:
            # Create train/valid directories
            train_images_dir: Path = images_dir / "train"
            train_labels_dir: Path = labels_dir / "train"
            valid_images_dir: Path = images_dir / "valid"
            valid_labels_dir: Path = labels_dir / "valid"

            for dir_path in [train_images_dir, train_labels_dir, valid_images_dir, valid_labels_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)

            result["train"] = (train_images_dir, train_labels_dir)
            result["valid"] = (valid_images_dir, valid_labels_dir)
        else:
            # Create directory for the specified split
            split_dir = split.lower()
            split_images_dir: Path = images_dir / split_dir
            split_labels_dir: Path = labels_dir / split_dir

            split_images_dir.mkdir(parents=True, exist_ok=True)
            split_labels_dir.mkdir(parents=True, exist_ok=True)

            result[split_dir] = (split_images_dir, split_labels_dir)

        return result

    def _check_training_dataset_exists(self, dataset_dir: Path) -> bool:
        """
        Legacy method, use _check_dataset_exists instead.
        """
        return self._check_dataset_exists(dataset_dir, Split.TRAIN)

    def _use_existing_training_dataset(self, dataset_dir: Path) -> str:
        """
        Legacy method, use _use_existing_dataset instead.
        """
        return self._use_existing_dataset(dataset_dir, Split.TRAIN)

    def _check_evaluation_dataset_exists(self, dataset_dir: Path) -> bool:
        """
        Legacy method, use _check_dataset_exists instead.
        """
        return self._check_dataset_exists(dataset_dir, Split.TEST)

    def _use_existing_evaluation_dataset(self, dataset_dir: Path) -> str:
        """
        Legacy method, use _use_existing_dataset instead.
        """
        return self._use_existing_dataset(dataset_dir, Split.TEST)

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
                dataset_version = dataforge.get_latest_dataset_version(
                    dataset_uuid=dataset_uuid,
                    split=split,
                    access_token=self.token_handler.tokens.access_token,
                )
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
            error_msg = f"Failed to get dataset version for UUID: {dataset_uuid}, split: {split}"
            logger.error(error_msg)
            raise DatasetNotFoundError(error_msg=error_msg)

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

    def _download_dataset(
        self,
        dataset_uuid: str,
        output_dir: Path,
        split: str,
        max_retries: int = 3,
        retry_delay: int = 5,
        verbose: bool = False,
    ) -> Tuple[DatasetVersionResponse, Path]:
        """
        Download dataset from DataForge.

        Args:
            dataset_uuid: The UUID of the dataset to download
            output_dir: Directory to save downloaded files
            split: Dataset split to download
            max_retries: Maximum number of retry attempts for network/storage errors
            retry_delay: Delay in seconds between retry attempts (will increase with each retry)
            verbose: Whether to log detailed progress for each file

        Returns:
            Tuple[DatasetVersionResponse, Path]: Tuple containing the dataset version response and temporary directory path

        Raises:
            DatasetNotFoundError: If the dataset with the given UUID is not found
            DatasetDownloadError: If the dataset download fails
        """
        try:
            # Create temporary directory for downloads
            temp_dir: Path = output_dir / "temp_download"
            temp_dir.mkdir(parents=True, exist_ok=True)

            # Get the dataset version with retry logic
            dataset_version = self._get_dataset_version_with_retry(dataset_uuid, split, max_retries, retry_delay)

            # Download data with retry logic
            if not self._download_dataset_with_retry(
                dataset_version, temp_dir, split, max_retries, retry_delay, verbose
            ):
                error_msg = f"Failed to download dataset for UUID: {dataset_uuid}, split: {split}"
                logger.error(error_msg)
                raise DatasetDownloadError(message=error_msg)

            return dataset_version, temp_dir

        except (DatasetNotFoundError, DatasetDownloadError):
            # Re-raise known exceptions
            raise
        except Exception as e:
            error_msg = f"Error downloading dataset: {str(e)}"
            logger.exception(error_msg)
            raise DatasetDownloadError(error_msg) from e

    def _copy_files_with_progress(
        self,
        split_pairs: List[Tuple[List[Tuple[Path, Path]], Path, Path]],
        total_pairs: int,
        is_for_training: bool,
        split: str,
        verbose: bool = False,
    ) -> Tuple[int, int, int]:
        """
        Copy files with progress tracking.

        Args:
            split_pairs: List of tuples containing (file_pairs, images_dest_dir, labels_dest_dir)
            total_pairs: Total number of file pairs to copy
            is_for_training: Whether this is for training dataset
            split: Dataset split name
            verbose: Whether to log detailed progress for each file

        Returns:
            Tuple[int, int, int]: Tuple containing (copy_success_count, copy_error_count, skipped_count)
        """
        # Initialize progress tracking
        progress_bar = tqdm(total=total_pairs, desc="Copying files", unit="files")

        # Set logging interval for large datasets
        log_interval: int = max(1, min(1000, total_pairs // 10))  # Log at most 10 times

        # Copy files with error handling
        copy_success_count: int = 0
        copy_error_count: int = 0
        skipped_count: int = 0
        processed_count: int = 0

        # Process each split
        for pairs, images_dest_dir, labels_dest_dir in split_pairs:
            for img_file, ann_file in pairs:
                try:
                    # For evaluation, check if files already exist
                    dest_img_file: Path = images_dest_dir / img_file.name
                    dest_ann_file: Path = labels_dest_dir / ann_file.name

                    if not is_for_training and dest_img_file.exists() and dest_ann_file.exists():
                        skipped_count += 1
                        if verbose:
                            logger.debug(f"Skipping existing file pair: {img_file.name}")
                    else:
                        img_success: bool = self._copy_file_with_error_handling(img_file, dest_img_file)
                        ann_success: bool = self._copy_file_with_error_handling(ann_file, dest_ann_file)

                        if img_success and ann_success:
                            copy_success_count += 1
                            if verbose:
                                split_name = "training" if images_dest_dir.name == "train" else (
                                    "validation" if images_dest_dir.name == "valid" else split.lower()
                                )
                                logger.debug(f"Copied {split_name} file: {img_file.name}")
                        else:
                            copy_error_count += 1
                except Exception as e:
                    copy_error_count += 1
                    logger.warning(f"Error processing file pair {img_file.name}: {str(e)}")

                processed_count += 1
                # Update progress
                progress_bar.update(1)
                if processed_count % log_interval == 0 or processed_count == total_pairs:
                    progress_pct: float = 100 * processed_count / total_pairs
                    logger.info(f"Progress: {processed_count}/{total_pairs} files processed ({progress_pct:.1f}%)")

        # Close progress bar
        progress_bar.close()

        # Log summary statistics
        if copy_error_count > 0:
            logger.warning(f"Encountered {copy_error_count} errors while copying files")
        if skipped_count > 0:
            logger.info(f"Skipped {skipped_count} already existing files")

        logger.success(f"Successfully copied {copy_success_count} files")

        return copy_success_count, copy_error_count, skipped_count

    def _prepare_dataset(
        self,
        dataset_version: DatasetVersionResponse,
        dataset_dir: Path,
        temp_dir: Path,
        dataset_uuid: str,
        split: str,
        valid_split: float = 0.1,
        random_seed: int = 0,
        verbose: bool = False,
    ) -> str:
        """
        Prepare downloaded dataset by organizing it into the proper directory structure.

        Args:
            dataset_version: Dataset version response
            dataset_dir: Directory where the dataset is being prepared
            temp_dir: Temporary directory containing downloaded files
            dataset_uuid: UUID of the dataset
            split: Dataset split
            valid_split: Ratio of validation data to split from train data (0.0-1.0), only used if split='TRAIN'
            random_seed: Random seed for reproducible splitting, only used if split='TRAIN'
            verbose: Whether to log detailed progress for each file

        Returns:
            str: Path to the configured dataset

        Raises:
            DatasetPrepareError: If the dataset preparation fails
        """
        try:
            # Save id_mapping
            self._save_id_mapping(dataset_version, dataset_dir / "id_mapping.json")

            # Get source file paths
            source_images_dir: Path = temp_dir / dataset_uuid / "images"
            source_annotations_dir: Path = temp_dir / dataset_uuid / "annotations"

            if not source_images_dir.exists() or not source_annotations_dir.exists():
                error_msg = "Required source directories not found after download"
                logger.error(error_msg)
                raise DatasetPrepareError(message=error_msg)

            # Find all valid image-annotation pairs
            file_pairs = self._find_file_pairs(source_images_dir, source_annotations_dir)

            if not file_pairs:
                error_msg = "No valid image-annotation pairs found in the dataset"
                logger.error(error_msg)
                raise DatasetPrepareError(message=error_msg)

            is_for_training = (split == Split.TRAIN)

            # Create directory structure based on split
            split_dirs = self._create_split_directories(dataset_dir, split)

            # Prepare pairs for each split
            if is_for_training:
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
                split_pairs = [
                    (train_pairs, *split_dirs["train"]),
                    (valid_pairs, *split_dirs["valid"])
                ]
            else:
                split_dir = split.lower()
                total_pairs: int = len(file_pairs)
                split_pairs = [(file_pairs, *split_dirs[split_dir])]

            # Copy files with progress tracking
            copy_success_count, copy_error_count, skipped_count = self._copy_files_with_progress(
                split_pairs=split_pairs,
                total_pairs=total_pairs,
                is_for_training=is_for_training,
                split=split,
                verbose=verbose
            )

            if copy_success_count == 0:
                error_msg = "Failed to copy any files during dataset preparation"
                logger.error(error_msg)
                raise DatasetPrepareError(message=error_msg)

            # Log final results
            if is_for_training:
                logger.success(f"Dataset downloaded, split and configured at: {dataset_dir}")
                valid_count = len(split_pairs[1][0]) if len(split_pairs) > 1 else 0
                train_count = len(split_pairs[0][0])
                logger.info(f"Train samples: {train_count}, Validation samples: {valid_count}")
            else:
                logger.success(f"Evaluation dataset downloaded at: {dataset_dir}")
                logger.info(f"Total evaluation samples: {copy_success_count + skipped_count}")

            return dataset_dir.as_posix()

        except DatasetPrepareError:
            # Re-raise known exceptions
            raise
        except Exception as e:
            error_msg = f"Error preparing dataset: {str(e)}"
            logger.exception(error_msg)
            raise DatasetPrepareError(error_msg) from e

    def _download_and_prepare_dataset(
        self,
        dataset_uuid: str,
        output_dir: str,
        split: str,
        valid_split: float = 0.1,
        random_seed: int = 0,
        max_retries: int = 3,
        retry_delay: int = 5,
        verbose: bool = False,
    ) -> str:
        """
        Common logic for downloading and preparing datasets for both training and evaluation.

        Args:
            dataset_uuid: The UUID of the dataset to download
            output_dir: Directory to save downloaded files
            split: Dataset split to download. If 'TRAIN', the dataset will be split into train/valid
            valid_split: Ratio of validation data to split from train data (0.0-1.0), only used if split='TRAIN'
            random_seed: Random seed for reproducible splitting, only used if split='TRAIN'
            max_retries: Maximum number of retry attempts for network/storage errors
            retry_delay: Delay in seconds between retry attempts (will increase with each retry)
            verbose: Whether to log detailed progress for each file

        Returns:
            str: Path to the configured dataset

        Raises:
            DatasetNotFoundError: If the dataset with the given UUID is not found
            DatasetDownloadError: If the dataset download fails
            DatasetPrepareError: If the dataset preparation fails
        """
        temp_dir = None
        try:
            # Create base output directory
            dataset_dir: Path = Path(output_dir) / dataset_uuid

            # Check if dataset already exists
            if self._check_dataset_exists(dataset_dir, split):
                return self._use_existing_dataset(dataset_dir, split)

            # Dataset doesn't exist or is incomplete, proceed with download
            dataset_dir.mkdir(parents=True, exist_ok=True)

            is_for_training = (split == Split.TRAIN)
            purpose = "training" if is_for_training else "evaluation"
            logger.info(f"Downloading {purpose} dataset with UUID: {dataset_uuid}{', split: ' + split if not is_for_training else ''}")

            # Step 1: Download the dataset
            dataset_version, temp_dir = self._download_dataset(
                dataset_uuid=dataset_uuid,
                output_dir=dataset_dir,
                split=split,
                max_retries=max_retries,
                retry_delay=retry_delay,
                verbose=verbose
            )

            # Step 2: Prepare the dataset
            result = self._prepare_dataset(
                dataset_version=dataset_version,
                dataset_dir=dataset_dir,
                temp_dir=temp_dir,
                dataset_uuid=dataset_uuid,
                split=split,
                valid_split=valid_split,
                random_seed=random_seed,
                verbose=verbose
            )

            return result

        except (DatasetNotFoundError, DatasetDownloadError, DatasetPrepareError):
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            purpose = "training" if split == Split.TRAIN else "evaluation"
            error_msg = f"Unexpected error in download_dataset_for_{purpose}: {str(e)}"
            logger.exception(error_msg)
            raise DatasetPrepareError(error_msg) from e
        finally:
            # Ensure cleanup happens even if an exception occurs
            if temp_dir is not None and temp_dir.exists():
                self._cleanup_temp_dir(temp_dir)

    def download_dataset_for_training(
        self,
        dataset_uuid: str,
        output_dir: str = "/datasets",
        valid_split: float = 0.1,
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

        Raises:
            DatasetNotFoundError: If the dataset with the given UUID is not found
            DatasetDownloadError: If the dataset download fails
            DatasetPrepareError: If the dataset preparation fails
        """
        return self._download_and_prepare_dataset(
            dataset_uuid=dataset_uuid,
            output_dir=output_dir,
            split=Split.TRAIN,
            valid_split=valid_split,
            random_seed=random_seed,
            max_retries=max_retries,
            retry_delay=retry_delay,
            verbose=verbose,
        )

    def download_dataset_for_evaluation(
        self,
        dataset_uuid: str,
        output_dir: str = "/datasets",
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

        Raises:
            DatasetNotFoundError: If the dataset with the given UUID is not found
            DatasetDownloadError: If the dataset download fails
            DatasetPrepareError: If the dataset preparation fails
        """
        return self._download_and_prepare_dataset(
            dataset_uuid=dataset_uuid,
            output_dir=output_dir,
            split=split,
            max_retries=max_retries,
            retry_delay=retry_delay,
            verbose=verbose,
        )

    def get_dataset_version_from_dataforge(self, dataset_uuid: str, split: Split) -> DatasetVersionInfo:
        dataset_version = self._get_dataset_version_with_retry(dataset_uuid, split)

        return dataset_version.data

    def get_dataset_info_from_dataforge(self, project_id: str, dataset_uuid: str, split: Split) -> DatasetPayload:
        dataset_info = dataforge.get_dataset(project_id, dataset_uuid, self.token_handler.tokens.access_token)

        return dataset_info.data
