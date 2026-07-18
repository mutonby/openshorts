"""Cloudflare R2 (S3-compatible) storage for the users' durable video library.

Layout: users/<user_id>/<job_id>/<filename>. Presigned URLs give private,
time-limited view/download links. Delete-by-prefix wipes a user's whole library
when their subscription's grace period ends.
"""
import boto3
from botocore.config import Config

from .config import settings

_client = None


def client():
    global _client
    if _client is None:
        _client = boto3.client(
            "s3",
            endpoint_url=settings.r2_endpoint,
            aws_access_key_id=settings.r2_access_key_id,
            aws_secret_access_key=settings.r2_secret_access_key,
            region_name="auto",
            config=Config(signature_version="s3v4", retries={"max_attempts": 3}),
        )
    return _client


def user_prefix(user_id) -> str:
    return f"users/{user_id}/"


def job_key(user_id, job_id, filename) -> str:
    return f"users/{user_id}/{job_id}/{filename}"


def upload_file(local_path, key, content_type="video/mp4"):
    client().upload_file(local_path, settings.r2_bucket, key,
                         ExtraArgs={"ContentType": content_type})


def download_file(key, local_path):
    client().download_file(settings.r2_bucket, key, local_path)


def delete_key(key):
    client().delete_object(Bucket=settings.r2_bucket, Key=key)


def list_keys(prefix) -> list:
    """List every object key under a prefix."""
    c = client()
    keys = []
    token = None
    while True:
        kw = {"Bucket": settings.r2_bucket, "Prefix": prefix, "MaxKeys": 1000}
        if token:
            kw["ContinuationToken"] = token
        resp = c.list_objects_v2(**kw)
        keys.extend(o["Key"] for o in resp.get("Contents", []))
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    return keys


def presigned_get(key, expires=3600, download_name=None) -> str:
    params = {"Bucket": settings.r2_bucket, "Key": key}
    if download_name:
        params["ResponseContentDisposition"] = f'attachment; filename="{download_name}"'
    return client().generate_presigned_url("get_object", Params=params, ExpiresIn=expires)


def delete_prefix(prefix) -> int:
    """Delete every object under a prefix. Returns the count deleted."""
    c = client()
    deleted = 0
    token = None
    while True:
        kw = {"Bucket": settings.r2_bucket, "Prefix": prefix, "MaxKeys": 1000}
        if token:
            kw["ContinuationToken"] = token
        resp = c.list_objects_v2(**kw)
        objs = [{"Key": o["Key"]} for o in resp.get("Contents", [])]
        if objs:
            c.delete_objects(Bucket=settings.r2_bucket, Delete={"Objects": objs})
            deleted += len(objs)
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    return deleted
