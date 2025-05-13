import csv
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple

from loguru import logger
from tqdm import tqdm

from netspresso.clients.dataforge.main import DataForgeClient
from netspresso.clients.dataforge.schemas.response_body import (
    DatasetResponse,
    DatasetsResponse,
    DatasetVersionResponse,
    DatasetVersionsResponse,
)
from netspresso.clients.dataforge.storage import S3Provider


class Split(str, Enum):
    TRAIN = "train"
    TEST = "test"


class DataForge:
    def __init__(self):
        self.client = DataForgeClient()
        self.s3 = S3Provider()
        # Default bucket names
        self.data_bucket = "rawdata"
        self.annotation_bucket = "annotations"
        self.csv_bucket = "dataset"

    def get_datasets(self, project_id: str, access_token: str) -> DatasetsResponse:
        """Get all datasets in a project"""
        return self.client.get_datasets(project_id, access_token)

    def get_dataset(self, project_id: str, dataset_uuid: str, access_token: str) -> DatasetResponse:
        """Get a specific dataset by UUID and split"""
        return self.client.get_dataset(project_id, dataset_uuid, access_token)

    def get_dataset_versions(self, dataset_uuid: str, split: str, access_token: str) -> DatasetVersionsResponse:
        """Get all versions of a dataset by UUID and split"""
        return self.client.get_dataset_versions(dataset_uuid, split, access_token)

    def get_latest_dataset_version(self, dataset_uuid: str, split: str, access_token: str) -> DatasetVersionResponse:
        """Get the latest version of a dataset by UUID and split"""
        return self.client.get_latest_dataset_version(dataset_uuid, split, access_token)

    def _download_csv_file(self, csv_path: str, dataset_dir: Path) -> bool:
        """
        Download CSV file from S3 bucket

        Args:
            csv_path: S3 path of the CSV file
            dataset_dir: Local directory to save the file

        Returns:
            bool: True if download successful, False otherwise
        """
        if not csv_path:
            logger.error("CSV path information is missing")
            return False

        # Create local CSV file path
        local_csv_path = dataset_dir / Path(csv_path).name

        # Download CSV file
        logger.info(f"Downloading CSV file: {self.csv_bucket}/{csv_path}")
        if not self.s3.download_file(bucket=self.csv_bucket, object_name=csv_path, dest_path=str(local_csv_path)):
            logger.error("CSV file download failed")
            return False

        logger.success(f"CSV file downloaded successfully: {local_csv_path}")
        return True

    def _create_or_load_status_file(self, csv_path: Path, status_csv_path: Path) -> List[Dict]:
        """
        Create or load a status file from CSV file.

        Args:
            csv_path: Original CSV file path
            status_csv_path: CSV file path for status tracking

        Returns:
            List[Dict]: List of rows containing status information
        """
        if status_csv_path.exists():
            # Load existing status file
            try:
                with open(status_csv_path, 'r', newline='') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)

                # Check if status file is not empty
                if rows:
                    logger.info(f"Loaded existing status file with {len(rows)} entries")

                    # Convert string boolean values to actual booleans
                    for row in rows:
                        row['data_downloaded'] = row.get('data_downloaded', '').lower() == 'true'
                        row['annotation_downloaded'] = row.get('annotation_downloaded', '').lower() == 'true'

                    return rows
                else:
                    logger.warning("Status file exists but is empty or invalid, creating new status file")
            except Exception as e:
                logger.warning(f"Error loading status file: {e}. Creating new status file.")

        # Create new status file
        try:
            # Load original CSV file
            rows = []
            try:
                with open(csv_path, 'r', newline='') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
            except Exception as e:
                logger.error(f"Error reading source CSV file: {e}")
                return []

            if not rows:
                logger.error("Source CSV file is empty or invalid")
                return []

            # Add download status fields
            for row in rows:
                row['data_downloaded'] = False
                row['annotation_downloaded'] = False
                row['timestamp'] = datetime.now().isoformat()

            # Save status file immediately (headers and initial state)
            self._update_status_file(status_csv_path, rows)
            logger.info(f"Created new status file with {len(rows)} entries")

            return rows

        except Exception as e:
            logger.error(f"Error creating status file: {e}")
            return []

    def _update_status_file(self, status_csv_path: Path, rows: List[Dict]) -> None:
        """
        Update the status file.

        Args:
            status_csv_path: Status CSV file path
            rows: List of row data to update
        """
        try:
            # Handle empty rows case
            if not rows:
                logger.warning("No rows to update in status file")
                return

            # Ensure all required fields exist in rows
            required_fields = ['data_downloaded', 'annotation_downloaded', 'timestamp']
            for field in required_fields:
                if field not in rows[0]:
                    logger.warning(f"Required field '{field}' missing in status data")
                    rows[0][field] = False if 'downloaded' in field else ''

            # Prepare field names
            fieldnames = list(rows[0].keys())

            # First write to temporary file then rename (atomic file writing)
            temp_file = status_csv_path.with_suffix('.tmp')
            try:
                with open(temp_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
                    f.flush()  # Flush buffer immediately

                # Verify temporary file was written successfully
                if not temp_file.exists() or temp_file.stat().st_size == 0:
                    raise IOError("Failed to write temporary status file")

                # Rename temporary file to actual file (atomic operation)
                temp_file.replace(status_csv_path)

            except (IOError, OSError) as e:
                logger.error(f"Error writing to temporary file: {e}")
                # Clean up temporary file
                if temp_file.exists():
                    try:
                        temp_file.unlink()
                        logger.debug("Successfully removed temporary file after error")
                    except OSError as unlink_error:
                        logger.warning(f"Failed to remove temporary file: {unlink_error}")
                raise

        except Exception as e:
            logger.error(f"Error updating status file: {e}")

    def _download_data_files(self, row: Dict, dataset_dir: Path) -> Tuple[bool, bool]:
        """
        Download data and annotation files for a single row

        Args:
            row: Dictionary containing file information
            dataset_dir: Directory to save downloaded files

        Returns:
            Tuple of (data_success, annotation_success)
        """
        # Get data and annotation file paths
        data_path = row.get('data_path')
        annotation_path = row.get('annotation_path')

        if not data_path or not annotation_path:
            logger.warning(f"Missing file path in row: {row}")
            return False, False

        # Set up local directory paths
        local_data_dir = dataset_dir / 'images'
        local_annotation_dir = dataset_dir / 'annotations'

        # Create directories
        local_data_dir.mkdir(parents=True, exist_ok=True)
        local_annotation_dir.mkdir(parents=True, exist_ok=True)

        # Extract filenames
        data_filename = Path(data_path).name
        annotation_filename = Path(annotation_path).name

        # Set up local file paths
        local_data_path = local_data_dir / data_filename
        local_annotation_path = local_annotation_dir / annotation_filename

        # Download files
        data_success = self.s3.download_file(
            bucket=self.data_bucket,
            object_name=data_path,
            dest_path=str(local_data_path)
        )

        annotation_success = self.s3.download_file(
            bucket=self.annotation_bucket,
            object_name=annotation_path,
            dest_path=str(local_annotation_path)
        )

        return data_success, annotation_success

    def download_dataset(self, dataset_version: DatasetVersionResponse, output_dir: str, verbose: bool = False) -> bool:
        """
        Download a dataset based on CSV information

        Args:
            dataset_version: Dataset version
            output_dir: Directory to save downloaded files
            verbose: Whether to log detailed progress for each file (default: False)

        Returns:
            bool: Whether the download was successful
        """
        rows = []  # Define globally to make accessible in exception blocks
        status_csv_path = None

        try:
            dataset_version_data = dataset_version.data

            # Create dataset directory path
            dataset_dir = Path(output_dir) / dataset_version_data.dataset_uuid
            dataset_dir.mkdir(parents=True, exist_ok=True)

            # Extract CSV path from metadata
            csv_path = dataset_version_data.dataset_metadata.csv_s3_path

            # Download CSV file
            if not self._download_csv_file(csv_path, dataset_dir):
                return False

            # Set local CSV path
            local_csv_path = dataset_dir / Path(csv_path).name

            # Create status CSV path and prepare status file
            status_csv_path = dataset_dir / f"{Path(csv_path).stem}_status.csv"
            rows = self._create_or_load_status_file(local_csv_path, status_csv_path)

            if not rows:
                logger.error("Failed to create or load status file")
                return False

            # Statistics counters
            success_count = 0
            failed_count = 0
            skipped_count = 0
            total_rows = len(rows)

            logger.info(f"Processing {total_rows} files")

            # Progress tracking setup
            try:
                use_progress_bar = True
                progress_bar = tqdm(total=total_rows, desc="Downloading files", unit="files")
            except ImportError:
                use_progress_bar = False
                logger.info("tqdm module not found, progress bar disabled")

            # Set logging interval - log status every X% of progress
            log_interval = max(1, min(500, total_rows // 20))  # Log at most 20 times for the entire dataset

            # Process each row in the status file
            for i, row in enumerate(rows, 1):
                try:
                    # Skip already downloaded files - now using boolean values directly
                    if row['data_downloaded'] and row['annotation_downloaded']:
                        if verbose:
                            logger.debug(f"[{i}/{total_rows}] Already downloaded: {row['data_path']}")
                        skipped_count += 1

                        # Update progress bar without logging
                        if use_progress_bar:
                            progress_bar.update(1)
                        continue

                    # Download files
                    if verbose:
                        logger.debug(f"[{i}/{total_rows}] Downloading file: {row.get('data_path')}")
                    data_success, annotation_success = self._download_data_files(row, dataset_dir)

                    # Update download status - store as boolean values
                    row['data_downloaded'] = data_success
                    row['annotation_downloaded'] = annotation_success
                    row['timestamp'] = datetime.now().isoformat()

                    # Track download results
                    if data_success and annotation_success:
                        success_count += 1
                    else:
                        failed_count += 1
                        # Always log failures regardless of verbose setting
                        logger.warning(f"[{i}/{total_rows}] Download failed - Data: {'success' if data_success else 'failed'}, "
                                      f"Annotation: {'success' if annotation_success else 'failed'}")

                    # Update status file after each download to enable resume capability
                    self._update_status_file(status_csv_path, rows)

                    # Update progress indicator
                    if use_progress_bar:
                        progress_bar.update(1)
                    elif i % log_interval == 0 or i == total_rows:
                        progress_pct = 100 * i / total_rows
                        logger.info(f"Progress: {i}/{total_rows} files processed ({progress_pct:.1f}%)")

                        # Also report current stats
                        if i % (log_interval * 5) == 0:  # Less frequent status updates
                            logger.info(f"Status: Success: {success_count}, Failed: {failed_count}, Skipped: {skipped_count}")

                except Exception as row_e:
                    logger.error(f"Error processing row {i}: {row_e}")
                    failed_count += 1
                    # Continue despite errors in individual rows
                    continue

            # Close progress bar if used
            if use_progress_bar:
                progress_bar.close()

            logger.success(f"Download complete - Success: {success_count}, Failed: {failed_count}, Skipped: {skipped_count}, Total: {total_rows}")
            return success_count > 0

        except KeyboardInterrupt:
            logger.warning("Download interrupted by user")
            # Final status file update on interrupt
            if status_csv_path and rows:
                try:
                    self._update_status_file(status_csv_path, rows)
                    logger.info("Status file updated before exit")
                except Exception as e:
                    logger.error(f"Failed to update status file on interrupt: {e}")
            return False

        except Exception as e:
            logger.exception(f"Error during dataset download: {e}")
            return False


dataforge = DataForge()
