import click

from redis import Redis
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
@click.option("--redis-host", default="redis")
@click.option("--job-queue", default="job-queue")
def run(redis_host: str, job_queue: str):
    redis = Redis(host=redis_host, port=6379, db=0)
    factory = Factory()
    while True:
        pending = redis.blpop(job_queue)
        task = factory.load(pending, TaskInfo)
        try:
            file_path = download(task.video_url, f"{task.id}.mp4")
            result = process_video(file_path)
            # TODO: Обработка
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
