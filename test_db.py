import asyncio

from psycopg_pool import AsyncConnectionPool

async def main():
    print("Testing pool")
    try:
        pool = AsyncConnectionPool("host=localhost dbname=design_db user=design_readonly password=read_password", open=False)
        await pool.open(timeout=2.0)
        print("SUCCESS")
    except Exception as e:
        print(f"FAILED: {e}")

asyncio.run(main())
