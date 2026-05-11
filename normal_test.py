import asyncio
import random


async def test(id):
    print("处理简单任务")
    print("遇到困难任务，等待执行")
    await asyncio.sleep(5)
    print(f"处理简单任务完成,{id}结束")


async def main():
    for i in range(3):
        print("等待tcp连接...")
        await asyncio.sleep(2)
        id = random.randint(1, 1000000)
        print(f"tcp连接成功:{id}")
        asyncio.create_task(test(id))




if __name__ == '__main__':
    # asyncio.get_event_loop().run_in_executor(None, lambda: main())
    asyncio.run(main())
