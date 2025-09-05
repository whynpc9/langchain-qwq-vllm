#!/usr/bin/env python3
"""
工具调用与结构化输出集成示例

这个示例展示了如何正确地结合工具调用和结构化输出功能。
基于对 vLLM 行为的深入测试，我们采用分阶段处理的方法。
"""

from langchain_core.tools import tool
from langchain_core.messages import ToolMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from langchain_qwq.chat_models_vllm import ChatQwenVllm


# 定义工具
@tool
def get_weather(location: str, unit: str = "celsius") -> str:
    """获取指定地点的当前天气信息.
    
    Args:
        location: 要查询天气的城市或地点
        unit: 温度单位，'celsius' 或 'fahrenheit'
    
    Returns:
        描述当前天气状况的字符串
    """
    # 模拟天气数据
    weather_data = {
        "beijing": {"temp": 22, "condition": "sunny", "humidity": 60},
        "shanghai": {"temp": 25, "condition": "cloudy", "humidity": 75},
        "tokyo": {"temp": 20, "condition": "sunny", "humidity": 55},
        "london": {"temp": 15, "condition": "rainy", "humidity": 80}
    }
    
    location_lower = location.lower()
    if location_lower in weather_data:
        data = weather_data[location_lower]
        temp_unit = "°C" if unit == "celsius" else "°F"
        if unit == "fahrenheit":
            data["temp"] = int(data["temp"] * 9/5 + 32)
        
        return f"当前{location}的天气：{data['condition']}，{data['temp']}{temp_unit}，湿度{data['humidity']}%"
    else:
        return f"暂无{location}的天气数据"


# 定义结构化输出模式
class WeatherQuery(BaseModel):
    """天气查询结果架构."""
    location: str = Field(description="查询的地点")
    temperature: int = Field(description="温度数值（不含单位）")
    condition: str = Field(description="天气状况")
    humidity: int = Field(description="湿度百分比", ge=0, le=100)
    unit: str = Field(description="温度单位", pattern="^(celsius|fahrenheit)$")


def basic_example():
    """基本示例：工具调用 + 结构化输出"""
    print("🌟 基本示例：工具调用 + 结构化输出")
    print("=" * 50)
    
    # 初始化 LLM
    llm = ChatQwenVllm(
        model="Qwen/Qwen3-32B",
        api_base="http://localhost:8000/v1",
        temperature=0.1,
    )
    
    # 阶段1: 工具调用（不使用 guided_json）
    print("📞 阶段1: 工具调用...")
    llm_with_tools = llm.bind_tools([get_weather])
    
    initial_messages = [
        SystemMessage(content="你是一个天气助手。使用可用的工具获取准确的天气信息。"),
        HumanMessage(content="东京的天气怎么样？")
    ]
    
    tool_response = llm_with_tools.invoke(initial_messages)
    print(f"   工具调用响应类型: {type(tool_response)}")
    
    if hasattr(tool_response, 'tool_calls') and tool_response.tool_calls:
        print(f"   生成了 {len(tool_response.tool_calls)} 个工具调用")
        
        # 执行工具调用
        tool_messages = []
        for tool_call in tool_response.tool_calls:
            if tool_call['name'] == 'get_weather':
                print(f"   执行工具: {tool_call['name']} with {tool_call['args']}")
                tool_result = get_weather.invoke(tool_call['args'])
                print(f"   工具结果: {tool_result}")
                
                tool_messages.append(
                    ToolMessage(
                        content=tool_result,
                        tool_call_id=tool_call['id']
                    )
                )
    else:
        print("   ⚠️ 没有生成工具调用")
        return
    
    # 阶段2: 结构化输出（使用 guided_json）
    print("\n📊 阶段2: 结构化输出...")
    structured_llm = llm.with_structured_output(
        schema=WeatherQuery,
        method="json_schema"
    )
    
    # 构建包含工具结果的完整对话
    conversation_with_tools = initial_messages + [tool_response] + tool_messages
    
    # 添加结构化输出指令
    final_messages = conversation_with_tools + [
        HumanMessage(content="""根据工具提供的天气信息，请按照 WeatherQuery 架构格式化响应：
1. 提取数值温度（不含单位符号）
2. unit 字段必须是 'celsius' 或 'fahrenheit'
3. 确保所有必需字段都已正确填写""")
    ]
    
    structured_response = structured_llm.invoke(final_messages)
    print(f"   结构化响应类型: {type(structured_response)}")
    
    if isinstance(structured_response, WeatherQuery):
        print("   ✅ 成功获得结构化数据:")
        print(f"      位置: {structured_response.location}")
        print(f"      温度: {structured_response.temperature}")
        print(f"      状况: {structured_response.condition}")
        print(f"      湿度: {structured_response.humidity}%")
        print(f"      单位: {structured_response.unit}")
    else:
        print(f"   ❌ 未获得预期的结构化数据: {structured_response}")


def include_raw_example():
    """高级示例：带原始响应的结构化输出"""
    print("\n🔍 高级示例：include_raw=True")
    print("=" * 50)
    
    llm = ChatQwenVllm(
        model="Qwen/Qwen3-32B", 
        api_base="http://localhost:8000/v1",
        temperature=0.1,
    )
    
    # 阶段1: 工具调用
    llm_with_tools = llm.bind_tools([get_weather])
    
    messages = [
        SystemMessage(content="你是天气分析师。使用工具获取多个城市的天气信息。"),
        HumanMessage(content="请查询北京和上海的天气。")
    ]
    
    tool_response = llm_with_tools.invoke(messages)
    
    # 执行所有工具调用
    tool_messages = []
    if hasattr(tool_response, 'tool_calls') and tool_response.tool_calls:
        for tool_call in tool_response.tool_calls:
            if tool_call['name'] == 'get_weather':
                result = get_weather.invoke(tool_call['args'])
                tool_messages.append(
                    ToolMessage(content=result, tool_call_id=tool_call['id'])
                )
    
    # 阶段2: 带原始响应的结构化输出
    class WeatherSummary(BaseModel):
        cities: list[str] = Field(description="查询的城市列表")
        overall_condition: str = Field(description="总体天气状况")
        temperature_range: str = Field(description="温度范围描述")
    
    structured_llm = llm.with_structured_output(
        schema=WeatherSummary,
        method="json_schema",
        include_raw=True
    )
    
    conversation = messages + [tool_response] + tool_messages
    final_messages = conversation + [
        HumanMessage(content="根据工具获取的天气信息，提供一个综合的天气摘要。")
    ]
    
    response = structured_llm.invoke(final_messages)
    
    if isinstance(response, dict) and "raw" in response:
        print("   ✅ 获得带原始响应的结构化数据:")
        print(f"   原始响应类型: {type(response['raw'])}")
        print(f"   解析状态: {'成功' if response['parsing_error'] is None else '失败'}")
        
        if response['parsed']:
            parsed = response['parsed']
            print(f"   城市: {parsed.cities}")
            print(f"   总体状况: {parsed.overall_condition}")
            print(f"   温度范围: {parsed.temperature_range}")


def main():
    """主函数"""
    print("🚀 vLLM 工具调用与结构化输出集成示例")
    print("基于分阶段处理的正确实现方式")
    print("=" * 60)
    
    try:
        basic_example()
        include_raw_example()
        
        print("\n" + "=" * 60)
        print("🎉 示例完成！")
        print("\n💡 关键要点:")
        print("1. 工具调用阶段：只发送 tools，不发送 guided_json")
        print("2. 结构化输出阶段：只发送 guided_json，不发送 tools")
        print("3. 通过对话历史传递工具调用结果")
        print("4. 两个阶段分别优化，确保最佳效果")
        
    except Exception as e:
        print(f"❌ 示例执行失败: {e}")
        print("💡 请确保 vLLM 服务器正在运行并加载了相应模型")


if __name__ == "__main__":
    main()
