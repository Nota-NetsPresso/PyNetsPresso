import argparse
import os
from datetime import datetime
from pathlib import Path

from loguru import logger

from netspresso.trainer.dataforge.dataforget import DataForge


def download_dataset_by_split(dataset_uuid: str, splits: list, output_dir: str):
    """
    Download datasets for specified dataset_uuid with multiple splits (train, test, etc.)
    and save them in separate folders by split.

    Args:
        dataset_uuid: Dataset UUID
        splits: List of splits to download (e.g., ['train', 'test'])
        output_dir: Base directory to save downloaded results
    """
    # Initialize DataForge client
    dataforge = DataForge()

    logger.info(f"Starting download for dataset UUID: {dataset_uuid}")
    logger.info(f"Processing splits: {', '.join(splits)}")

    # Create base directory (dataset_uuid folder)
    base_dir = Path(output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    # Track download results
    success_splits = []
    failed_splits = []

    for split in splits:
        # Create output directory for each split
        split_output_dir = base_dir / split
        split_output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"=== Starting download for {split} dataset ===")
        logger.info(f"Save path: {split_output_dir}")

        # Check latest dataset version information
        dataset_version = dataforge.get_latest_dataset_version(dataset_uuid, split)
        if not dataset_version or not dataset_version.data:
            logger.error(f"Could not get dataset information for {split} split")
            failed_splits.append(split)
            continue

        # Display metadata information
        metadata = dataset_version.data.dataset_metadata
        logger.info("Dataset information:")
        logger.info(f"- Bucket: {metadata.dataset_bucket_name}")
        logger.info(f"- Type: {metadata.dataset_type}")
        if hasattr(metadata, 'csv_s3_path'):
            logger.info(f"- CSV path: {metadata.csv_s3_path}")

        # Extract CSV path
        csv_path = dataset_version.data.dataset_metadata.csv_s3_path

        # Download and process CSV file directly
        csv_download_result = dataforge._download_csv_file(csv_path, split_output_dir)
        if not csv_download_result:
            logger.error(f"CSV file download failed for {split}")
            failed_splits.append(split)
            continue

        # Get data list from CSV file
        local_csv_path = split_output_dir / Path(csv_path).name
        status_csv_path = split_output_dir / f"{Path(csv_path).stem}_status.csv"

        # Create status file and process data download
        rows = dataforge._create_or_load_status_file(local_csv_path, status_csv_path)

        # Download actual data files
        success_count = 0
        failed_count = 0
        skipped_count = 0
        total_rows = len(rows)

        logger.info(f"Processing {total_rows} files for {split}")

        # Process each file
        for i, row in enumerate(rows, 1):
            if row['data_downloaded'] and row['annotation_downloaded']:
                logger.info(f"[{i}/{total_rows}] Already downloaded: {row['data_path']}")
                skipped_count += 1
                continue

            # Download files
            logger.info(f"[{i}/{total_rows}] Downloading file: {row.get('data_path')}")
            data_success, annotation_success = dataforge._download_data_files(row, split_output_dir)

            # Update status
            row['data_downloaded'] = data_success
            row['annotation_downloaded'] = annotation_success
            row['timestamp'] = datetime.now().isoformat()

            # Track results
            if data_success and annotation_success:
                success_count += 1
            else:
                failed_count += 1
                logger.warning(f"Download failed - Data: {'success' if data_success else 'failed'}, "
                                f"Annotation: {'success' if annotation_success else 'failed'}")

            # Update status file
            dataforge._update_status_file(status_csv_path, rows)

        logger.success(f"{split} download complete - Success: {success_count}, Failed: {failed_count}, Skipped: {skipped_count}")

        if success_count > 0:
            success_splits.append(split)
        else:
            failed_splits.append(split)

    # Result summary
    logger.info("\n=== Download Results Summary ===")
    logger.info(f"Successful splits: {', '.join(success_splits) if success_splits else 'None'}")
    logger.info(f"Failed splits: {', '.join(failed_splits) if failed_splits else 'None'}")

    return len(success_splits) > 0


def test_specific_dataset():
    """Function to test a specific dataset"""
    # Test dataset information
    dataset_uuid = "54c7c552-8974-4333-a81f-bf82f830f3bf"  # Dataset UUID to test
    splits = ["train", "test"]  # List of splits to download
    output_dir = f"./datasets/{dataset_uuid}"  # Save path

    # Execute dataset download
    result = download_dataset_by_split(dataset_uuid, splits, output_dir)

    if result:
        logger.success("Test completed: Dataset download successful")
    else:
        logger.error("Test failed: Error occurred during dataset download")

    return result


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="DataForge dataset download test")
    parser.add_argument("--dataset-uuid", type=str, help="Dataset UUID to download")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "test"],
                        help="List of splits to download (default: train test)")
    parser.add_argument("--output-dir", type=str, default="./datasets",
                        help="Directory to save download results (default: ./datasets)")

    args = parser.parse_args()

    if args.dataset_uuid:
        # Download dataset received from command line arguments
        output_dir = os.path.join(args.output_dir, args.dataset_uuid)
        download_dataset_by_split(args.dataset_uuid, args.splits, output_dir)
    else:
        # Download predefined test dataset
        test_specific_dataset()


if __name__ == "__main__":
    main()
