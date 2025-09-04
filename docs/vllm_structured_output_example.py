#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example: Using ChatQwenVllm with structured output

This example demonstrates how to use ChatQwenVllm's with_structured_output()
method to get structured JSON responses using vLLM's guided_json parameter.
"""

from langchain_qwq import ChatQwenVllm
from pydantic import BaseModel
from typing import List


# Define your schema as a Pydantic model
class OperationAnalysis(BaseModel):
    """Schema for medical operation analysis."""
    code: str
    name: str
    match_flag: str  # "完全", "部分", or "未匹配"
    score: int  # 0 for no match, 40-70 for partial, 70-100 for complete
    rules: str


class OperationResults(BaseModel):
    """Container for multiple operation analyses."""
    opers: List[OperationAnalysis]


def main():
    # Initialize ChatQwenVllm (requires running vLLM server)
    llm = ChatQwenVllm(
        model="Qwen/Qwen3-32B",
        api_base="http://localhost:8000/v1",
        temperature=0.0,
        max_tokens=32768,
    )
    
    # Create structured output chain
    # Note: Only "json_schema" method is supported for vLLM backend
    structured_llm = llm.with_structured_output(
        schema=OperationResults,
        method="json_schema"
    )
    
    # Example messages for medical coding analysis
    messages = [
        {
            "role": "system", 
            "content": "你是一名资深病案编码员，请分析手术记录与ICD编码的匹配程度。"
        },
        {
            "role": "user",
            "content": """
手术文本: 1:穿刺右桡动脉成功后置6F桡动脉鞘 2:循导丝送药物涂层球囊扩张
ICD编码: 1. 00.6600x008 经皮冠状动脉药物球囊扩张成形术 2. 00.4000 单根血管操作
            """
        }
    ]
    
    # Invoke the structured output
    try:
        result = structured_llm.invoke(messages)
        print("Structured Output Result:")
        for oper in result.opers:
            print(f"Code: {oper.code}")
            print(f"Name: {oper.name}")
            print(f"Match Flag: {oper.match_flag}")
            print(f"Score: {oper.score}")
            print(f"Rules: {oper.rules}")
            print("---")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your vLLM server is running at http://localhost:8000")


if __name__ == "__main__":
    main()
