import os
from decimal import Decimal
from typing import Optional

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, NoCredentialsError
from fastapi import HTTPException


class ObjectStorageHandler:
    def __init__(self) -> None:
        try:
            client_params = {
                "service_name": "s3",
                "aws_access_key_id": os.environ.get("SCALITY_ACCESS_KEY_ID"),
                "aws_secret_access_key": os.environ.get("SCALITY_SECRET_ACCESS_KEY"),
                "config": Config(
                    region_name="ap-northeast-2",
                    signature_version="s3v4",
                    retries={"max_attempts": 10, "mode": "standard"},
                ),
            }
            client_params["endpoint_url"] = os.environ.get("ZENKO_SERVER_URL")
            self.s3_client = boto3.client(**client_params)
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
