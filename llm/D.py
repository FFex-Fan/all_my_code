import re
import os
from openai import OpenAI

os.environ["OPENAI_API_KEY"] = "sk-O0EY9oxGBIYxCgK09aF3381e5a5d4b95A8C5C5463a61B2C7"
os.environ["OPENAI_BASE_URL"] = "https://gtapi.xiaoerchaoren.com:8932/v1"


# 创建第二个代理的策略（保持不变）
def agent2_strategy():
    return 'confess'


# 使用 GPT-3.5 作为第一个代理
def gpt_agent(scenario, last_decision):
    if last_decision is None:
        message = [
            {"role": "system", "content": scenario},
            {"role": "user",
             "content": "You are in the first round of the game. "
                        "Your task is to choose to confess or remain silent."
                        "Make your decision based on the classic 'Prisoner's Dilemma' game theory."
                        "Please choose one and provide your choice. "
                        "Your choice must be 'I choose to confess' or 'I choose to be remain silent.', no other irrelevant options can be printed."
             }
        ]
    else:
        past_decisions = ', '.join(f"Round {i + 1}: '{decision}'" for i, decision in enumerate(last_decision))
        message = [
            {"role": "system", "content": scenario},
            {"role": "user",
             "content": f"In previous rounds, your partner's decisions were: {past_decisions}. "
                        "Now it's your turn again. Do you change your strategy, "
                        "or do you stick to your original choice? "
                        "Make your decision based on the classic 'Prisoner's Dilemma' game theory."
                        "Please choose one and provide your choice. "
                        "Your choice must be 'I choose to confess' or 'I choose to be remain silent.', no other irrelevant options can be printed."}
        ]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=message,
        temperature=2.0,
        max_tokens=150
    )
    decision = response.choices[0].message.content
    return decision


# 正则匹配选择的内容
def get_decision(agent1_decision, agent2_decision):
    decision = None
    pattern = r"I choose to (\b\w+(?:\s\w+)?\b)"
    match = re.search(pattern, agent1_decision)
    if match:
        # The decision is captured in the first group of the regex
        decision = match.group(1)
    else:
        decision = "No decision found"
    print("Extracted decision:", decision)
    return decision, agent2_decision


# 定义一个简单的 moderator 函数
def simple_moderator(agent1_decision, agent2_decision):
    # 初始化支付变量
    agent1_payoff, agent2_payoff = None, None

    # 解析代理的决策并计算支付
    if agent1_decision == 'confess' and agent2_decision == 'confess':
        agent1_payoff, agent2_payoff = -5, -5
    elif agent1_decision == 'confess' and agent2_decision == 'remain silent':
        agent1_payoff, agent2_payoff = 0, -10
    elif agent1_decision == 'remain silent' and agent2_decision == 'confess':
        agent1_payoff, agent2_payoff = -10, 0
    elif agent1_decision == 'remain silent' and agent2_decision == 'remain silent':
        agent1_payoff, agent2_payoff = -1, -1  # Both serve 1 year

    return agent1_payoff, agent2_payoff


client = OpenAI()
# 定义囚徒困境游戏的场景描述
scenario = """You and your partner have been arrested and placed in separate isolation cells. 
You can't communicate with your partner, but you both have been offered a deal. 
If you confess and your partner remains silent, you'll be set free while your partner will serve 10 years in prison. 
If both of you confess, you'll both serve 5 years. 
If both of you remain silent, you'll both serve only 1 year."""

last_decision = []

# 循环进行多轮游戏
for round in range(10):  # 你可以选择进行多少轮游戏
    # 第一个代理的决策（使用 GPT-3.5）
    agent1_decision = gpt_agent(scenario, last_decision)

    # 第二个代理的决策（保持不变）
    agent2_decision = agent2_strategy()
    last_decision.append(agent2_decision)
    print(last_decision)

    # 获取两个代理的决定
    agent1_decision1, agent2_decision1 = get_decision(agent1_decision, agent2_decision)
    # 调解者判断并计算收益
    agent1_payoff, agent2_payoff = simple_moderator(agent1_decision1, agent2_decision1)
    # 打印每一轮的决策结果
    print(f"Round {round + 1}: Agent 1 decision - {agent1_decision}, Agent 2 decision - {agent2_decision}")
    print(f"Payoffs: Agent 1 - {agent1_payoff}, Agent 2 - {agent2_payoff}")
    print()
