import re
from typing import List, Dict, Any


def mock_parse_action_part(action_part_content: str) -> List[Dict[str, Any]]:
    """
    为了方便测试，我提取了你函数中的核心逻辑
    """
    # 这里使用了你原本的正则
    action_iterator = re.finditer(r'Action: (.*?)(?=\s*(\n+|Action:|\Z))', action_part_content, re.S)

    tool_call_list = []
    for action_match in action_iterator:
        action_content = action_match.group(1).strip()

        if action_content.startswith("Finish"):
            # 模拟 Finish 解析
            try:
                finish_answer = action_content.split("[")[1].split("]")[0].strip()
                print(f"  [检测到 Finish]: {finish_answer}")
                # 注意：你原代码这里直接 return [] 会导致前面的工具被丢弃
                # 测试时我们先记录下来
            except:
                print(f"  [Finish 解析失败]: {action_content}")
            continue

        else:
            # 模拟工具解析
            try:
                # 注意：你原代码这里的 $ 会对末尾空格敏感
                tool_match = re.search(r'(\w+)\[(.*?)\]$', action_content)
                if tool_match:
                    tool_name = tool_match.group(1)
                    tool_args_content = tool_match.group(2)
                    tool_params = dict(re.findall(r'(\w+)="(.*?)"', tool_args_content))
                    tool_call_list.append({
                        "tool_name": tool_name,
                        "tool_params": tool_params
                    })
                else:
                    print(f"  [正则未命中]: {action_content}")
            except Exception as e:
                print(f"  [解析异常]: {action_content} -> {e}")

    return tool_call_list


# --- 测试用例设计 ---
test_cases = [
    {
        "name": "标准单工具调用",
        "input": 'Action: calculate[expression="123*456"]'
    },
    {
        "name": "标准双工具调用",
        "input": 'Action: get_weather[city="上海"] Action: calculate[expression="1+1"]'
    },
    {
        "name": "带空格的工具调用 (容易失败)",
        "input": 'Action: calculate[expression="1+1"] '  # 注意末尾有个空格
    },
    {
        "name": "跨行的 Action (需要 re.S)",
        "input": 'Action: calculate[\nexpression="1+1"\n]'
    },
    {
        "name": "Finish 在最后",
        "input": 'Action: calculate[expression="1+1"] Action: Finish[任务完成]'
    },
    {
        "name": "追加了额外的东西",
        "input": 'Action: get_weather[city="上海"] Action: calculate[expression="1+1"]  \n(等待回复结果)'
    }

]

print("=== 开始解析测试 ===\n")
for case in test_cases:
    print(f"测试点: {case['name']}")
    print(f"输入内容: {repr(case['input'])}")
    result = mock_parse_action_part(case['input'])
    print(f"解析结果: {result}")
    print("-" * 50)