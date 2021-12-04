import click
import boto3
import redis

from typing import Optional
from dataclass_factory import Factory
from loguru import logger
from pathlib import Path

from .models import TaskInfo
from .utils import download
from .processing import process_video


@click.group()
def cli():
    pass


@cli.command()
@click.option("--redis-url", default="redis://redis:6379/0")
@click.option("--job-queue", default="job-queue")
@click.option("--minio-access-key")
@click.option("--minio-secret-key")
@click.option("--minio-endpoint", default="http://localhost:9000")
@click.option()
def run(redis_url: str, job_queue: str, minio_access_key: str, minio_secret_key: str, minio_endpoint: str):
    r = redis.from_url(redis_url)
    s3client = boto3.client(
        service_name='s3',
        aws_access_key_id=minio_access_key,
        aws_secret_access_key=minio_secret_key,
        config=boto3.session.Config(signature_version='s3v4'),
        endpoint_url=minio_endpoint
    )

    factory = Factory()
    while True:
        pending = r.blpop(job_queue)
        task = factory.load(pending, TaskInfo)
        try:
            file_path = download(task.video_url, f"{task.id}.mp4")
            result_path = process_video(file_path)
            s3client.upload_file(result_path, "processed-videos", f"{task.id}.mp4")
            r.set(f"task:{task.id}", 1)
        except:  # noqa
            logger.exception(f"Error during processing task {task.id}!")


@cli.command()
@click.option("--video-url")
@click.option("--video-path", type=click.Path(exists=True, dir_okay=False))
def process(video_url: Optional[str] = None, video_path: Optional[str] = None):
    if video_url and not video_path:
        video_path = download(video_url, "video.mp4")
    video_path = Path(video_path)
    result = process_video(video_path)
    print(result)


if __name__ == '__main__':
    cli(auto_envvar_prefix="WORKER")
