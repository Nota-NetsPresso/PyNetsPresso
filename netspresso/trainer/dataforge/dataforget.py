import csv
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple

from loguru import logger

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

    def get_datasets(self, project_id: str) -> DatasetsResponse:
        """Get all datasets in a project"""
        return self.client.get_datasets(project_id)

    def get_dataset(self, dataset_uuid: str, split: str) -> DatasetResponse:
        """Get a specific dataset by UUID and split"""
        return self.client.get_dataset(dataset_uuid, split)

    def get_dataset_versions(self, dataset_uuid: str, split: str) -> DatasetVersionsResponse:
        """Get all versions of a dataset by UUID and split"""
        return self.client.get_dataset_versions(dataset_uuid, split)

    def get_latest_dataset_version(self, dataset_uuid: str, split: str) -> DatasetVersionResponse:
        """Get the latest version of a dataset by UUID and split"""
        return self.client.get_latest_dataset_version(dataset_uuid, split)

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

    def _create_or_load_status_file(self, local_csv_path: Path, status_csv_path: Path) -> List[Dict]:
        """
        Create a new status file or load an existing one

        Args:
            local_csv_path: Path to the original CSV file
            status_csv_path: Path to the status CSV file

        Returns:
            List of dictionaries containing row data with status information
        """
        # Check if status file already exists
        if status_csv_path.exists():
            logger.info(f"Using existing status file: {status_csv_path}")
            with open(status_csv_path, 'r') as status_file:
                rows = list(csv.DictReader(status_file))
                # Convert string 'True'/'False' to boolean values
                for row in rows:
                    row['data_downloaded'] = row['data_downloaded'].lower() == 'true'
                    row['annotation_downloaded'] = row['annotation_downloaded'].lower() == 'true'
                return rows

        # Create new status file
        logger.info(f"Creating new status file: {status_csv_path}")

        # Read original CSV
        with open(local_csv_path, 'r') as original, open(status_csv_path, 'w', newline='') as status_file:
            reader = csv.DictReader(original)
            fieldnames = reader.fieldnames + ['data_downloaded', 'annotation_downloaded', 'timestamp']
            writer = csv.DictWriter(status_file, fieldnames=fieldnames)
            writer.writeheader()

            rows = []
            for row in reader:
                status_row = dict(row)
                status_row['data_downloaded'] = False  # Boolean instead of string
                status_row['annotation_downloaded'] = False  # Boolean instead of string
                status_row['timestamp'] = ''
                # Need to convert to string for CSV writing
                writer.writerow({
                    **status_row,
                    'data_downloaded': str(status_row['data_downloaded']),
                    'annotation_downloaded': str(status_row['annotation_downloaded'])
                })
                rows.append(status_row)

        return rows

    def _update_status_file(self, status_csv_path: Path, rows: List[Dict]) -> None:
        """
        Update the status file with the current download status

        Args:
            status_csv_path: Path to the status CSV file
            rows: List of dictionaries containing row data with updated status
        """
        if not rows:
            return

        with open(status_csv_path, 'w', newline='') as status_file:
            fieldnames = rows[0].keys()
            writer = csv.DictWriter(status_file, fieldnames=fieldnames)
            writer.writeheader()

            # Convert boolean values to strings for CSV writing
            for row in rows:
                csv_row = dict(row)
                csv_row['data_downloaded'] = str(row['data_downloaded'])
                csv_row['annotation_downloaded'] = str(row['annotation_downloaded'])
                writer.writerow(csv_row)

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

    def download_dataset(self, dataset_version: DatasetVersionResponse, output_dir: str) -> bool:
        """
        Download a dataset based on CSV information

        Args:
            dataset_version: Dataset version
            output_dir: Directory to save downloaded files

        Returns:
            bool: Whether the download was successful
        """
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

            # Statistics counters
            success_count = 0
            failed_count = 0
            skipped_count = 0
            total_rows = len(rows)

            logger.info(f"Processing {total_rows} files")

            # Process each row in the status file
            for i, row in enumerate(rows, 1):
                # Skip already downloaded files - now using boolean values directly
                if row['data_downloaded'] and row['annotation_downloaded']:
                    logger.info(f"[{i}/{total_rows}] Already downloaded: {row['data_path']}")
                    skipped_count += 1
                    continue

                # Download files
                logger.info(f"[{i}/{total_rows}] Downloading file: {row.get('data_path')}")
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
                    logger.warning(f"Download failed - Data: {'success' if data_success else 'failed'}, "
                                  f"Annotation: {'success' if annotation_success else 'failed'}")

                # Update status file after each download to enable resume capability
                self._update_status_file(status_csv_path, rows)

            logger.success(f"Download complete - Success: {success_count}, Failed: {failed_count}, Skipped: {skipped_count}, Total: {total_rows}")
            return success_count > 0

        except Exception as e:
            logger.exception(f"Error during dataset download: {e}")
            return False


dataforge = DataForge()
