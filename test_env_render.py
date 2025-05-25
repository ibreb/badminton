from Env_class_court_step import Env  # 假设你的Env类在env.py中
from rule_player import Player
from datetime import datetime
import os
import csv 

def csv_to_env_history(csv_path):
    # 动作名称到索引的映射
    action_mapping = {
        '发短球': 0, '长球': 1, '推球': 2, '杀球': 3, '挡小球': 4,
        '平球': 5, '放小球': 6, '挑球': 7, '切球': 8, '发长球': 9
    }
    
    history = {}
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            # 转换字段类型
            step = int(row['Step'])
            current_player = int(row['Current_Player'])
            losing_player = int(row['Losing_Player']) if row['Losing_Player'] != '-1' else -1
            
            # 构建state元组 (Player0_X, Player0_Y, Player1_X, Player1_Y, Ball_X, Ball_Y, Ball_Height, Current_Player)
            state = (
                int(float(row['Player0_X'])),  # 坐标转整数
                int(float(row['Player0_Y'])),
                int(float(row['Player1_X'])),
                int(float(row['Player1_Y'])),
                float(row['Ball_X']),
                int(float(row['Ball_Y'])),
                int(float(row['Ball_Height'])),
                current_player
            )
            
            # 构建action元组 (动作索引, 落点坐标, 高度, 结果状态)
            action_type = action_mapping.get(row['Action_Type'], 10)  # 未知动作映射到索引10
            action = (
                action_type,
                (float(row['Landing_X']), int(row['Landing_Y'])),
                int(row['Action_Height']),
                'success' if row['Failure_Reason'] == '' else row['Failure_Reason']
            )
            
            # 构建next_state元组
            next_state = (
                int(float(row['Next_Player0_X'])),
                int(float(row['Next_Player0_Y'])),
                int(float(row['Next_Player1_X'])),
                int(float(row['Next_Player1_Y'])),
                float(row['Next_Ball_X']),
                int(float(row['Next_Ball_Y'])),
                int(float(row['Next_Ball_Height'])),
                int(row['Next_Current_Player'])
            )
            
            # 组合完整记录
            history[step] = {
                'state': state,
                'action': action,
                'next_state': next_state,
                'reward': int(row['Reward']),
                'score_player0': int(row['Score_Player0']),
                'score_player1': int(row['Score_Player1']),
                'failure_reason': row['Failure_Reason'],
                'losing_player': losing_player
            }
    
    return history

def main():
    # 初始化部分不变
    player0 = Player(0)
    player1 = Player(1)
    env = Env(player0, player1)
    state = env.reset()
    
    print("===== 比赛开始 =====")
    
    while True:
        print(f"\n当前比分：玩家 {env.score_player0} - {env.score_player1} 对手")
        
        # 新增发球阶段重置
        env.is_serve_phase = True  # 确保每分开始都是发球阶段
        current_serve_player = env.current_player  # 记录当前发球方
        
        round_done = False
        while not round_done:
            current_player = env.current_player
            player = player0 if current_player == 0 else player1
            
            # 发球阶段验证
            if env.is_serve_phase:
                if current_player != current_serve_player:
                    print("⚠️ 错误：发球方与得分方不匹配！")
                    return
                
                action = player.serve(state)
                print(f"发球方：Player {current_player} 发球到 {action[1]}")
            else:
                action = player.generate_shot(state)
                print(f"Player {current_player} 回球到 {action[1]} 方式 {env.ACTIONS[action[0]]}")
            
            next_state, reward, done, info = env.step(action)
            
            env.save_to_csv("match_recording.csv")
            
            print("比赛记录已保存")
            state = next_state
            
            if done:
                # 关键修复：重置发球阶段和发球权
                env.is_serve_phase = True  # 为下一分准备
                env.current_player = info['losing_player']  # 失败方下一分继续发球？
                
                # 应改为：由得分方发球
                winner = 0 if env.score_player0 > env.score_player1 else 1
                env.current_player = winner  # 得分方获得发球权
                
                
                print(f"★ 本分结束：{info['failure_reason']} 失分方：Player {info['losing_player']}")
                print(f"最新比分：玩家 {env.score_player0} - {env.score_player1} 对手")
                #state = env.reset() 
                # 胜负判定
                if max(env.score_player0, env.score_player1) >= 21:
                    print("\n===== 比赛结束 =====")
                    # 保存和渲染代码...
                    env.history = csv_to_env_history("match_recording.csv")  
                    env.render()
                    return
                state = env.reset() #重置发球方
                break  # 结束当前分循环


if __name__ == "__main__":
    main()
