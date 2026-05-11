import asyncio

from src.api.dependencies.common import get_settings
from src.api.services.storage.r2 import R2StorageService, R2StorageSettings


async def main():
    s = get_settings()
    r2 = R2StorageService(
        settings=R2StorageSettings(
            account_id=s.r2_account_id,
            access_key_id=s.r2_access_key_id,
            secret_access_key=s.r2_secret_access_key,
            bucket_name=s.r2_bucket_name,
        )
    )
    async with r2._get_client() as client:
        resp = await client.list_objects_v2(Bucket=s.r2_bucket_name, MaxKeys=1)
        print("r2 ok:", resp.get("Name"))


asyncio.run(main())
