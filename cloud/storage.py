"""Cloudflare R2 (S3-compatible) storage for the users' durable video library.

Layout: users/<user_id>/<job_id>/<filename>, plus a paid job's source video at
users/<user_id>/<job_id>/source/<random>.<ext> (cloud/source_archive.py), kept
out of project restores. Presigned URLs give private,
time-limited view/download links. Delete-by-prefix wipes a user's whole library
when their subscription's grace period ends.
"""
from urllib.parse import quote

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


def upload_fileobj(fileobj, key, content_type="video/mp4", metadata=None):
    """Upload from a file handle the caller already holds open.

    ``upload_file`` re-opens the path for every multipart part, so a disk sweep
    that deletes the file halfway through a large upload breaks it. A handle
    opened once keeps reading the same inode whatever happens to the name. The
    price is that parts are buffered in memory, hence two parts at a time
    instead of the default ten.
    """
    from boto3.s3.transfer import TransferConfig
    extra = {"ContentType": content_type}
    if metadata:
        extra["Metadata"] = {str(k): str(v) for k, v in metadata.items()}
    # Only options boto3's own TransferConfig takes: s3transfer's has more
    # (max_in_memory_upload_chunks), and passing one of those here raised on
    # every upload.
    config = TransferConfig(max_concurrency=2)
    client().upload_fileobj(fileobj, settings.r2_bucket, key, ExtraArgs=extra, Config=config)


def download_file(key, local_path):
    client().download_file(settings.r2_bucket, key, local_path)


def object_metadata(key):
    """The user metadata of one object (``x-amz-meta-*``), or None when the
    object is not there."""
    try:
        head = client().head_object(Bucket=settings.r2_bucket, Key=key)
    except client().exceptions.ClientError as e:
        code = str((e.response or {}).get("Error", {}).get("Code", ""))
        if code in ("404", "NoSuchKey", "NotFound"):
            return None
        raise
    return dict(head.get("Metadata") or {})


def object_size(key):
    """Size in bytes of one object, or None when it is not there."""
    try:
        return int(client().head_object(Bucket=settings.r2_bucket, Key=key)["ContentLength"])
    except client().exceptions.ClientError as e:
        code = str((e.response or {}).get("Error", {}).get("Code", ""))
        if code in ("404", "NoSuchKey", "NotFound"):
            return None
        raise


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


def iter_objects(prefix):
    """Yield ``(key, last_modified, size)`` for every object under a prefix,
    one listing page at a time, so a sweep over the whole bucket never holds
    the full key list in memory."""
    c = client()
    token = None
    while True:
        kw = {"Bucket": settings.r2_bucket, "Prefix": prefix, "MaxKeys": 1000}
        if token:
            kw["ContinuationToken"] = token
        resp = c.list_objects_v2(**kw)
        for o in resp.get("Contents", []):
            yield o["Key"], o.get("LastModified"), int(o.get("Size") or 0)
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")


def delete_keys(keys) -> int:
    """Delete many objects, 1000 per request. Returns the count sent."""
    keys = list(keys)
    c = client()
    for i in range(0, len(keys), 1000):
        c.delete_objects(Bucket=settings.r2_bucket,
                         Delete={"Objects": [{"Key": k} for k in keys[i:i + 1000]]})
    return len(keys)


def presigned_get(key, expires=3600, download_name=None) -> str:
    """A URL a client can GET. Signed against the S3 endpoint by default; a plain
    URL on the bucket's custom domain when R2_PUBLIC_BASE is set.

    Switching here rather than at each call site on purpose: every caller hands
    the result to a browser, and the S3 endpoint is the one variant browsers
    cannot reliably fetch (see settings.r2_public_base). The custom domain has no
    equivalent of ResponseContentDisposition, so a download keeps the object's
    own name, which is already the clip filename.
    """
    base = settings.r2_public_base
    if base:
        return f"{base}/{quote(key, safe='/')}"
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
