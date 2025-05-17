from decimal import Decimal
from typing import Optional

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, NoCredentialsError
from fastapi import HTTPException
from loguru import logger

from app.configs.settings import settings


class ObjectStorageHandler:
    def __init__(self) -> None:
        try:
            client_params = {
                "service_name": "s3",
                "aws_access_key_id": settings.SCALITY_ACCESS_KEY_ID,
                "aws_secret_access_key": settings.SCALITY_SECRET_ACCESS_KEY,
                "config": Config(
                    region_name="ap-northeast-2",
                    signature_version="s3v4",
                    retries={"max_attempts": 10, "mode": "standard"},
                ),
            }
            client_params["endpoint_url"] = settings.ZENKO_SERVER_URL
            self.s3_client = boto3.client(**client_params)

            # Ensure required buckets exist
            required_buckets = ["model", "evaluation"]
            for bucket in required_buckets:
                self._ensure_bucket_exists(bucket)

        except NoCredentialsError:
            raise HTTPException(
                status_code=500,
                detail="Failed to initialize S3 client: Missing credentials"
            )

    def get_download_presigned_url(
        self,
        bucket_name: str,
        object_path: str,
        download_name: Optional[str] = None,
        expires_in: int = 43200,
    ) -> str:
        """Generate presigned URL for downloading an object

        Args:
            bucket_name: S3 bucket name
            object_path: Path to object in bucket
            download_name: Optional filename for download
            expires_in: URL expiration time in seconds

        Returns:
            str: Presigned download URL

        Raises:
            HTTPException: If URL generation fails
        """
        try:
            params = {
                "Bucket": bucket_name,
                "Key": object_path,
            }
            if download_name is not None:
                params["ResponseContentDisposition"] = f'attachment; filename="{download_name}"'

            return self.s3_client.generate_presigned_url(
                "get_object",
                Params=params,
                ExpiresIn=expires_in,
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate download URL: {str(e)}"
            )

    def get_upload_presigned_url(
        self,
        bucket_name: str,
        object_path: str,
        expires_in: int = 43200,
    ):
        params = {
            "Bucket": bucket_name,
            "Key": object_path,
        }
        response = self.s3_client.generate_presigned_url(
            "put_object",
            Params=params,
            ExpiresIn=expires_in,
        )
        return response

    def check_file_exists(
        self,
        bucket_name: str,
        object_path: str,
    ) -> bool:
        try:
            self.s3_client.head_object(Bucket=bucket_name, Key=object_path)
            return True
        except ClientError as e:
            if e.response["Error"]["Message"] == "Not Found":
                return False

    def get_total_size_in_mb(
        self,
        bucket_name: str,
        object_path: str,
    ) -> float:
        """Get total size of objects under prefix in MB

        Args:
            bucket_name: S3 bucket name
            object_path: Object path prefix

        Returns:
            float: Total size in MB
        """
        try:
            total_size = 0
            response = self.s3_client.list_objects_v2(Bucket=bucket_name, Prefix=object_path)

            for obj in response.get("Contents", []):
                total_size += obj["Size"]

            total_size = total_size / (1024 * 1024)  # Convert to MB
            return float(str(Decimal(str(total_size))))

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to get object size: {str(e)}"
            )

    def copy_s3_object(
        self,
        bucket_name,
        source_object_path,
        destination_object_path,
    ):
        self.s3_client.copy(
            {
                "Bucket": bucket_name,
                "Key": source_object_path,
            },
            bucket_name,
            destination_object_path,
        )

    def get_head_object(
        self,
        bucket_name: str,
        object_path: str,
    ):
        return self.s3_client.head_object(Bucket=bucket_name, Key=object_path)

    def create_bucket(self, bucket_name: str):
        return self.s3_client.create_bucket(Bucket=bucket_name)

    def _ensure_bucket_exists(self, bucket_name: str) -> bool:
        """
        Check if a bucket exists and create it if it doesn't.

        Args:
            bucket_name: Name of the bucket to check/create

        Returns:
            bool: True if bucket exists or was created successfully

        Raises:
            HTTPException: If bucket creation fails
        """
        try:
            # Check if bucket exists
            logger.info(f"Checking if bucket {bucket_name} exists")
            self.s3_client.head_bucket(Bucket=bucket_name)
            return True
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code')

            # If bucket doesn't exist (404) or we don't have permission to check (403)
            if error_code == '404' or error_code == '403':
                try:
                    # Try to create the bucket
                    self.create_bucket(bucket_name)
                    logger.info(f"Bucket {bucket_name} created successfully")
                    return True
                except Exception as create_error:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to create required bucket {bucket_name}: {str(create_error)}"
                    )
            else:
                # Other unexpected error
                raise HTTPException(
                    status_code=500,
                    detail=f"Error checking bucket {bucket_name}: {str(e)}"
                )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected error checking bucket {bucket_name}: {str(e)}"
            )

    def upload_file_to_s3(
        self,
        bucket_name: str,
        local_path: str,
        object_path: str,
    ):
        try:
            self.s3_client.upload_file(local_path, bucket_name, object_path)
        except FileNotFoundError:
            raise Exception("The file was not found")

    def download_file_from_s3(
        self,
        bucket_name: str,
        local_path: str,
        object_path: str,
    ):
        try:
            self.s3_client.download_file(bucket_name, object_path, local_path)
        except FileNotFoundError:
            raise Exception("The file was not found")

    def list_objects(
        self,
        bucket_name: str,
        prefix: str,
    ) -> list:
        """List objects in a bucket with specified prefix

        Args:
            bucket_name: S3 bucket name
            prefix: Object prefix to filter results

        Returns:
            list: List of object paths (keys)

        Raises:
            HTTPException: If operation fails
        """
        try:
            result = []
            paginator = self.s3_client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(
                Bucket=bucket_name,
                Prefix=prefix
            )

            for page in page_iterator:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        result.append(obj['Key'])

            return result
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to list objects: {str(e)}"
            )
