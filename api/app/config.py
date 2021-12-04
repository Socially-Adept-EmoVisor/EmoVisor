from pydantic import BaseSettings


class Settings(BaseSettings):
    redis_url: str = "redis://redis:6379/0"
    minio_endpoint: str = "http://localhost:9000"
    minio_access_key: str = "api"
    minio_secret_key: str = "pikM8TSzU6n4pScP"
    job_queue: str = "job-queue"


settings = Settings()
