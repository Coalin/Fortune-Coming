# test_deepseek.py
import os
import sys
from openai import OpenAI

import os
from openai import OpenAI

def call_workflow_api(user_message, system_prompt='', stream=False):
    """
    调用DeepSeek API
    
    Args:
        system_prompt: 系统提示，如"You are a helpful assistant"
        user_message: 用户消息，如"Hello"
        stream: 是否流式输出
    """
    client = OpenAI(api_key='sk-dbaa7a9ad0414eab9d784c3b9a90d824', base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        stream=False
    )

    return response.choices[0].message.content

def test_basic():
    """测试基本功能"""
    print("=== 测试1: 基本调用 ===")
    try:
        result = call_workflow_api("用中文说一句问候语")
        print(f"回复: {result}")
        print("✅ 测试通过")
    except Exception as e:
        print(f"❌ 测试失败: {e}")

def test_system_prompt():
    """测试系统提示"""
    print("\n=== 测试2: 带系统提示 ===")
    try:
        result = call_workflow_api(
            "1+1等于多少？",
            system_prompt="你是一个数学老师，请详细解释"
        )
        print(f"回复: {result[:50]}...")
        print("✅ 测试通过")
    except Exception as e:
        print(f"❌ 测试失败: {e}")

def test_stream():
    """测试流式输出"""
    print("\n=== 测试3: 流式输出 ===")
    try:
        print("流式回复: ", end="")
        for chunk in call_workflow_api("简单介绍一下Python", stream=True):
            print(chunk, end="", flush=True)
        print("\n✅ 测试通过")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")

def test_custom_model():
    """测试自定义模型"""
    print("\n=== 测试4: 自定义模型 ===")
    try:
        result = call_workflow_api(
            "你好",
            model="deepseek-chat",
            api_key=os.environ.get('DEEPSEEK_API_KEY')
        )
        print(f"回复: {result[:30]}...")
        print("✅ 测试通过")
    except Exception as e:
        print(f"❌ 测试失败: {e}")

if __name__ == "__main__":
    # 检查API密钥
    api_key = os.environ.get('DEEPSEEK_API_KEY')
    if not api_key:
        print("⚠️ 警告: 未设置DEEPSEEK_API_KEY环境变量")
        print("请先运行: export DEEPSEEK_API_KEY='your-api-key'")
        sys.exit(1)
    
    print(f"API密钥已设置: {api_key[:10]}...\n")
    
    # 运行所有测试
    test_basic()
    
    print("\n=== 所有测试完成 ===")