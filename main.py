import asyncio

from pydantic import BaseModel
from langchain_qwq import ChatQwen
from langchain_core.tools import tool
from dotenv import load_dotenv
import os

os.environ["DASHSCOPE_API_BASE"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"


@tool
def get_weather(city: str) -> str:
    """Get the weather of a city"""
    return f"The weather of {city} is sunny"


model = ChatQwen(model="qwen3-32b").bind_tools([get_weather])


# async def call():
#     print(await model.ainvoke("What is the weather of Xian?"))


# async def main():
#     task1 = asyncio.create_task(call())
#     task2 = asyncio.create_task(call())
#     task3 = asyncio.create_task(call())
#     task4 = asyncio.create_task(call())
#     task5 = asyncio.create_task(call())
#     task6 = asyncio.create_task(call())
#     task7 = asyncio.create_task(call())
#     task8 = asyncio.create_task(call())
#     task9 = asyncio.create_task(call())
#     task10 = asyncio.create_task(call())
#     await asyncio.gather(
#         task1, task2, task3, task4, task5, task6, task7, task8, task9, task10
#     )


# if __name__ == "__main__":
#     asyncio.run(main())


async def call():
    class Man(BaseModel):
        name: str
        age: int

    model = ChatQwen(model="qwen3-32b").with_structured_output(
        Man, method="function_calling"
    )
    print(model.invoke("My Name is John and I am 20 years old"))


if __name__ == "__main__":
    asyncio.run(call())
