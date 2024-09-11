import random
import json

# operations = ['+', '-', '*', '/']
operations = ['+', '-']
max_num = 100000


def generate_arithmetic_problem():
    operation = random.choice(operations)

    if operation in ['+', '-', '*']:
        num1 = random.randint(0, max_num)
        num2 = random.randint(0, max_num)
    else:  # 除法
        num2 = random.randint(1, max_num)  # 限制除数范围，避免结果过于复杂
        num1 = num2 * random.randint(1, max_num)  # 确保能整除

    question = f"{num1} {operation} {num2} = ?"

    if operation == '+':
        answer = num1 + num2
    elif operation == '-':
        answer = num1 - num2
    elif operation == '*':
        answer = num1 * num2
    else:
        answer = num1 / num2  # 整数除法

    text = f"{num1} {operation} {num2} = {answer}"
    return {
        "prompt": question,
        "answer": str(answer),
        "text": text,
    }


def generate_training_data(num_problems):
    data = []
    for _ in range(num_problems):
        problem_dict = generate_arithmetic_problem()
        data.append(json.dumps(problem_dict, ensure_ascii=False))
    return "\n".join(data)


# 生成1000个问题的训练数据
training_data = generate_training_data(10 ** 5)

# 将数据保存到文件
with open("arithmetic_training_data.jsonl", "w", encoding="utf-8") as f:
    f.write(training_data)

print("训练数据已生成并保存到 arithmetic_training_data.jsonl")
