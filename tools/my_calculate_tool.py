import ast
import math
import operator


def my_calculate_tool(expression: str) -> str:
    """数学计算函数"""
    if expression is None or expression == "":
        return "计算式不能为空"

    # 支持的基本运算
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }

    # 支持的基本函数
    functions = {
        "sqrt": math.sqrt,
        "pi": math.pi
    }
    try:
        node = ast.parse(expression, mode="eval")
        result = _eval_node(node.body, operators, functions)
        return str(result)
    except:
        return "计算错误,请检查表达式格式"


def _eval_node(node, operators, functions):
    """递归计算节点的值"""
    # 2+math.sqrt((3/4))
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.BinOp):
        left = _eval_node(node.left, operators, functions)
        right = _eval_node(node.right, operators, functions)
        operator0 = operators[type(node.op)]
        return operator0(left, right)
    elif isinstance(node, ast.Call):
        function_name = node.func.attr
        if function_name in functions:
            args = [_eval_node(arg, operators, functions) for arg in node.args]
            return functions[function_name](*args)
    elif isinstance(node, ast.Name):
        if node.id in functions:
            return functions[node.id]
