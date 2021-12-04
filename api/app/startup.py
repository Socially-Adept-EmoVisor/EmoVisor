import contextlib

import aioboto3
import aioredis
from aiobotocore.session import ClientCreatorContext

from .main import app
from .config import settings


@app.on_event("startup")
async def init_redis():
    app.state.redis = await aioredis.from_url(settings.redis_url, max_connections=20)


@app.on_event("shutdown")
async def close_redis():
    await app.state.redis.close()


@app.on_event("startup")
async def init_boto():
    session = aioboto3.Session(
        aws_access_key_id=settings.minio_access_key,
        aws_secret_access_key=settings.minio_secret_key,
    )

    context_stack = contextlib.AsyncExitStack()
    s3client = await context_stack.enter_async_context(
        session.client("s3", endpoint_url=settings.minio_endpoint)
    )

    app.state.s3client = s3client
    app.state.context_stack = context_stack


@app.on_event("shutdown")
async def close_boto():
    await app.state.context_stack.aclose()
