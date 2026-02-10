import asyncio
import os
from redis.asyncio import Redis
import config.settings as settings

async def list_keys():
    redis_host = os.getenv('REDIS_HOST', 'localhost')
    redis_port = int(os.getenv('REDIS_PORT', 6379))
    redis_db = 0
    redis_url = f"redis://{redis_host}:{redis_port}/{redis_db}"
    print(f"Connecting to {redis_url}...")
    
    try:
        r = Redis.from_url(redis_url)
        keys = await r.keys("*")
        print(f"Found {len(keys)} keys:")
        for k in keys:
            print(f" - {k.decode('utf-8')}")
            
            # If it looks like a checkpoint, show type
            if b"checkpoint" in k:
                try:
                    val = await r.get(k)
                    print(f"   Value type: {type(val)} - Len: {len(val) if val else 0}")
                    # print(f"   Value: {val[:100]}") 
                except:
                    pass
        
        await r.aclose()
        
    except Exception as e:
        print(f"FAILED: {e}")

if __name__ == "__main__":
    asyncio.run(list_keys())
