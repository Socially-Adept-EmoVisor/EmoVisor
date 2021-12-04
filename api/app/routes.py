import re
from enum import IntEnum
from uuid import uuid4
from typing import Optional

from fastapi import Header
from fastapi import Depends
from fastapi import Request
from fastapi import APIRouter
from fastapi import HTTPException
from aioredis import Redis
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from botocore.exceptions import ClientError

from .config import settings

router = APIRouter()
CHUNK_SIZE = 1024 * 1024


def depends_redis(request: Request) -> Redis:
    return request.app.state.redis


def depends_s3(request: Request):
    return request.app.state.s3client


class ProcessRequest(BaseModel):
    source: str


class TaskInfo(BaseModel):
    id: str
    video_url: str


class ProcessResponse(BaseModel):
    task_id: str


class TaskStatus(IntEnum):
    PROCESSING = 0
    COMPLETED = 1


class StatusResponse(BaseModel):
    status: TaskStatus
    detail: str


class ByteRange(BaseModel):
    start: int
    end: int
    length: int


@router.post("/process", response_model=ProcessResponse)
async def test(data: ProcessRequest, redis: Redis = Depends(depends_redis)):
    task = TaskInfo(id=str(uuid4()), video_url=data.source)

    await redis.set(f"task:{task.id}", TaskStatus.PROCESSING)
    await redis.rpush(settings.job_queue, task.json())
    return ProcessResponse(task_id=task.id)


@router.get("/task_status", response_model=StatusResponse)
async def task_status(task_id: str, redis: Redis = Depends(depends_redis)):
    status = await redis.get(f"task:{task_id}")
    if status is None:
        raise HTTPException(detail="Task expired or does not exists.", status_code=400)

    status = int(status)
    if status == TaskStatus.PROCESSING:
        return StatusResponse(status=status, detail="In progress.")
    elif status == TaskStatus.COMPLETED:
        return StatusResponse(status=status, detail="Completed.")
    else:
        raise HTTPException(detail="Invalid status code.", status_code=500)


def parse_range_header(header: Optional[str]):
    if header is None:
        return 0, None
    match = re.fullmatch(r"bytes=(\d+)-(\d*)", header)
    if not match:
        return 0, None
    groups = match.groups()
    return [int(i) if i else None for i in groups]


@router.get("/video/{media_key}")
async def stream_media(
    media_key: str,
    data_range: Optional[str] = Header(None, alias="Range"),
    s3=Depends(depends_s3),
):
    if not media_key.endswith(".mp4"):
        raise HTTPException(detail="Not found.", status_code=404)

    try:
        media = await s3.get_object(Bucket="processed-videos", Key=media_key)
    except ClientError:
        raise HTTPException(detail="Not found.", status_code=404)

    content_length = media["ContentLength"]
    start, stop = parse_range_header(data_range)
    if start == 0 and stop is None:
        return StreamingResponse(
            media["Body"].iter_chunks(),
            media_type="video/mp4",
            headers={"Content-Length": str(content_length)},
        )

    if stop is None:
        stop = content_length - 1
    stop = max(content_length - 1, stop)

    media = await s3.get_object(
        Bucket="processed-videos", Key=media_key, Range=f"bytes={start}-{stop}"
    )

    return StreamingResponse(
        media["Body"].iter_chunks(chunk_size=CHUNK_SIZE),
        status_code=206,
        media_type="video/mp4",
        headers={
            "Content-Range": f"bytes {start}-{stop}/{content_length}",
            "Accept-Ranges": "bytes",
            "Content-Transfer-Encoding": "binary",
            "Connection": "Keep-Alive",
            "Content-Length": str(media["ContentLength"]),
        },
    )
