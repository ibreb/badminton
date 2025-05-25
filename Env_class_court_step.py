#2025-3-4修改了失误时的渲染代码,out\miss\hit net 都可以显示
#修改了miss时候的错误判断
#修改了换发球代码
# Env_10_9 同代码，作为class被引用
#修改为等比例真实场地 
import numpy as np
import os
from gym import spaces
from typing import Tuple, Literal, Dict, List
from Opponent_9_17 import Opponent  # 您的对手类，如果需要可以取消注释
import csv 
from matplotlib.animation import FuncAnimation
from rule_player import Player

import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.rcParams['font.family'] = 'Heiti TC'  # 替换为您选择的字体

class Env:

    def __init__(self, player0: Player, player1: Player):
        # 定义状态索引常量（建议放在类中）
        players = [player0, player1]
        #self.serving_player = 0  # 当前发球方 (0:玩家, 1:对手)
        self.current_player = 0 # 当前击球球员

        self.is_serve_phase = True
        self.history = []
        self.state_history = [] #存储3个state状态
        self.ACTIONS = ['发短球', '长球', '推球', '杀球', '挡小球', '平球', '放小球', '挑球', '切球', '发长球']
        # self.max_iteration = max_iteration
        self.state = (1, 1, 4, 1, 1, 1, 0, 0, 1)
        self.count = 0
        self.round_count = 0
        #self.all = 0
        #self.is_launch = True
        # self.prev_player_coord = (1, 1)
        self.shot_space = spaces.Discrete(11)
        self.action_space = spaces.Tuple((
            spaces.Discrete(11),
            spaces.MultiDiscrete([6, 3]),
            spaces.Discrete(3)
        ))
        self.observation_space = spaces.MultiDiscrete([3, 3, 3, 3, 6, 3, 3, 2])
        self._max_episode_steps = 1000
        #self.history = {}
        self.end_reason = "success"
        self.is_serve_phase = True
        self.last_winner = None

        # 添加分数记录
        self.score_player0 = 0
        self.score_player1 = 0
        self.winning_score = 10  # 比赛结束的分数

    def reset(self):
        #重置发球
        player_pos = (1, 1)
        opponent_pos = (4, 1)
        if self.current_player == 0:
            ball_pos = (1, 1)
        else:
            ball_pos = (4, 1)
        
        ball_height = 0

        self.state = [player_pos[0], player_pos[1], opponent_pos[0], opponent_pos[1], ball_pos[0], ball_pos[1], ball_height, self.current_player]
        #self.prev_player_coord = player_pos
        # self.history = []
        self.end_reason = ""
        self.round_count = 0  # 重置回合计数器



        # 重置分数
        # self.scores = [0, 0]
        # self.score_player0 = 0
        # self.score_player1 = 0

        self.count = 0  # 重置计数器

        return self.state


    def step(self, action):
        # 状态列表结构定义（按顺序存储元素）
        # state结构: [p0_x, p0_y, p1_x, p1_y, ball_x, ball_y, ball_h]
        P0_X, P0_Y, P1_X, P1_Y, BALL_X, BALL_Y, BALL_H = 0, 1, 2, 3, 4, 5, 6
        current_state = self.state.copy()  # 假设self.state是列表类型
        next_state = current_state.copy()  # 深拷贝当前状态
        reward = 0
        done = False
        failure_reason = "success"
        losing_player = -1
                



        step_info = {
            'state': current_state.copy(),
            'action': action.copy() if isinstance(action, list) else list(action),
            'next_state': None,
            'reward': reward,
            'score_player0': self.score_player0,
            'score_player1': self.score_player1,
            'failure_reason': failure_reason,
            'losing_player': losing_player
        }
        if len(self.state_history) >= 3:
            self.state_history.pop(0)
        self.state_history.append(self.state.copy())

        if self.is_serve_phase:
            # 发球动作生成（返回列表结构）
            current_player = self.current_player
            serve_action = action # [shot_type, landing_pos, height]
            
            
            # 更新球员位置（列表直接修改）
            if current_player == 0:
                # 发球方位置固定到(1,1)
                next_state[P0_X:P0_Y+1] = [1, 1]
                # 对手移动到落点（landing_pos是坐标列表）
                next_state[P1_X:P1_Y+1] = action[1]
                
            else:
                next_state[P1_X:P1_Y+1] = [4, 1]
                next_state[P0_X:P0_Y+1] = action[1]
            
            # 更新球状态（列表直接赋值）
            next_state[BALL_X:BALL_Y+1] = action[1]
            next_state[BALL_H] = action[2]
            next_state[-1] = 1 - current_player  # 保存当前击球方
            self.current_player = next_state[-1] # 更新当前击球方
            self.is_serve_phase = False
            step_info['action'] = ['发球']   # 列表合并

        else:
            # 非发球阶段击球动作验证（action为列表结构）
            failure_reason = self._check_shot_failure(action)
            current_player = self.current_player # 当前击球方
            

            if failure_reason == "success":
                # 解析动作参数
                shot_type = action[0]
                target_pos = action[1]  # 落地位置列表
                hit_height = action[2]
                
                # 更新球员位置（列表切片操作）
                if current_player == 0:
                    next_state[P0_X:P0_Y+1] = [1, 1]  # 击球方复位
                    next_state[P1_X:P1_Y+1] = target_pos  # 对手移动
                else:
                    next_state[P1_X:P1_Y+1] = [4, 1]
                    next_state[P0_X:P0_Y+1] = target_pos
                
                # 更新状态
                next_state[BALL_X:BALL_Y+1] = target_pos
                next_state[BALL_H] = hit_height
                next_state[-1] = 1 - current_player  # 保存当前击球方
                self.current_player = next_state[-1] # 更新击球方

            else:
                shot_type = action[0]
                target_pos = action[1]  # 落地位置列表
                hit_height = action[2]
                
                
                # 失败处理（示例：挂网）
                if failure_reason == "挂网":
                    done = True
                    # 重置双方位置
                    next_state[P0_X:P0_Y+1] = [1, 1]
                    next_state[P1_X:P1_Y+1] = [4, 1]
                    # 球位置重置到网中间
                    next_state[BALL_X:BALL_Y+1] = [2.5, 1]
                    
                    
                    next_state[BALL_H] = -1
                    
                    
                    # 得分处理
                    if current_player == 0:
                        self.score_player1 += 1
                        losing_player = 0
                        reward = -1
                        self.current_player = 1 #发球方
                    else:
                        self.score_player0 += 1
                        losing_player = 1
                        reward = -1
                        self.current_player = 0
                    #print("current_player",self.current_player)
                
                # 对手落地致胜
                elif failure_reason == "对手落地致胜":
                    done = True
                    if current_player == 0:
                        next_state[P0_X:P0_Y+1] = (1, 1)
                        next_state[P1_X:P1_Y+1] = target_pos
                        next_state[BALL_X:BALL_Y+1] = target_pos
                        next_state[BALL_H] = hit_height
                        self.score_player0 += 1
                        losing_player = 1
                        reward = 1  # 得分奖励
                        self.current_player = 0 # 下一分的发球方
                    else:
                        next_state[P1_X:P1_Y+1] = (4, 1)
                        next_state[P0_X:P0_Y+1] = target_pos
                        next_state[BALL_X:BALL_Y+1] = target_pos
                        next_state[BALL_H] = hit_height
                        self.score_player1 += 1
                        losing_player = 0
                        reward = 1 
                        self.current_player = 1 # 下一分的发球方
                    #print("current_player",self.current_player)
                # 出界处理
                else:
                    done = True
                    if current_player == 0:
                        next_state[P0_X:P0_Y+1] = (1, 1)
                        next_state[P1_X:P1_Y+1] = (4, 0)
                        # next_state['ball_pos'] = (4, -1)
                        # next_state['ball_height'] = 1
                        self.score_player1 += 1
                        losing_player = 0
                        reward = -1
                        self.current_player = 1 # 下一分的发球方
                    else:
                        next_state[P1_X:P1_Y+1] = (4, 1)
                        next_state[P0_X:P0_Y+1] = (1, 2)
                        self.score_player0 += 1
                        losing_player = 1
                        reward = -1
                        self.current_player = 0 # 下一分的发球方
                    
                    next_state[BALL_X:BALL_Y+1] = target_pos
                    next_state[BALL_H] = hit_height
                    #print("current_player",self.current_player)

            step_info.update({
                'failure_reason': failure_reason,
                'losing_player': losing_player
            })
        
        # 更新全局状态
        step_info['next_state'] = next_state.copy()
        step_info['reward'] = reward
        # 记录当前步信息
        step_record = self._create_step_record(
            current_state=current_state,
            action=action,
            next_state=next_state,
            step_info=step_info
        )
        self.history.append(step_record)
        self.state = next_state.copy()
        
        
        return self.state, reward, done, step_info
    
    def _create_step_record(self, current_state, action, next_state, step_info) -> Dict:
        """构建单步数据记录"""
        return {
            # 当前状态
            "current_p0_x": current_state[0],
            "current_p0_y": current_state[1],
            "current_p1_x": current_state[2],
            "current_p1_y": current_state[3],
            "current_ball_x": current_state[4],
            "current_ball_y": current_state[5],
            "current_ball_h": current_state[6],
            "current_player": current_state[7],
            
            # 动作信息
            "action_type_idx": action[0],
            "landing_x": action[1][0],
            "landing_y": action[1][1],
            "action_height": action[2],
            
            # 下一状态
            "next_p0_x": next_state[0],
            "next_p0_y": next_state[1],
            "next_p1_x": next_state[2],
            "next_p1_y": next_state[3],
            "next_ball_x": next_state[4],
            "next_ball_y": next_state[5],
            "next_ball_h": next_state[6],
            "next_player": next_state[7],
            
            # 结果信息
            "reward": step_info['reward'],
            "score_p0": step_info['score_player0'],
            "score_p1": step_info['score_player1'],
            "failure_reason": step_info['failure_reason'],
            "losing_player": step_info['losing_player']
        }

    def save_to_csv(self, filename: str):
        """保存历史记录到CSV文件"""
        headers = [
            'Step', 'Player0_X', 'Player0_Y', 'Player1_X', 'Player1_Y',
            'Ball_X', 'Ball_Y', 'Ball_Height', 'Current_Player',
            'Action_Type', 'Landing_X', 'Landing_Y', 'Action_Height',
            'Next_Player0_X', 'Next_Player0_Y', 'Next_Player1_X', 'Next_Player1_Y',
            'Next_Ball_X', 'Next_Ball_Y', 'Next_Ball_Height', 'Next_Current_Player',
            'Reward', 'Score_Player0', 'Score_Player1', 'Failure_Reason', 'Losing_Player'
        ]
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            
            for step_idx, record in enumerate(self.history):
                # 转换动作类型为中文
                #print("record[action_type_idx]",record["action_type_idx"])
                action_type = self.ACTIONS[record["action_type_idx"]] \
                    if int(record["action_type_idx"]) < len(self.ACTIONS) \
                    else "未知"
                
                # 构建CSV行数据
                row = [
                    step_idx,  # Step
                    # 当前状态
                    record["current_p0_x"], record["current_p0_y"],
                    record["current_p1_x"], record["current_p1_y"],
                    record["current_ball_x"], record["current_ball_y"],
                    record["current_ball_h"],
                    record["current_player"],
                    # 动作信息
                    action_type,
                    record["landing_x"], record["landing_y"],
                    record["action_height"],
                    # 下一状态
                    record["next_p0_x"], record["next_p0_y"],
                    record["next_p1_x"], record["next_p1_y"],
                    record["next_ball_x"], record["next_ball_y"],
                    record["next_ball_h"],
                    record["next_player"],
                    # 结果信息
                    record["reward"],
                    record["score_p0"], record["score_p1"],
                    record["failure_reason"],
                    record["losing_player"]
                ]
                writer.writerow(row)
    # Helper Methods
    # def _generate_serve(self):
    #     """生成发球动作参数"""
    #     if self.current_player == 0:
    #         # 玩家0发球到对方场地右侧（坐标[4,2]）
    #         return ('短球', (4, 2), 1)  # (shot_type, landing_pos, height)
    #     else:
    #         # 玩家1发球到对方场地左侧（坐标[1,2]）
    #         return ('短球', (1, 2), 1)

    def _check_shot_failure(self, action):
        """简化的击球失败检查（根据动作类型返回预设结果）"""
        shot_type = action[0]
        
        # 定义动作类型与结果的映射
        failure_map = {
            1: "success",
            2: "挂网",
            3: "对手落地致胜",
            4: "出界",
            5: "success"
        }
        
        # 返回对应的失败原因
        return failure_map.get(shot_type, "未知错误")



    def _check_game_over(self):
        """
        检查是否有一方达到胜利条件
        """
        if (self.score_player0 >= self.winning_score or self.score_player1 >= self.winning_score) and \
           abs(self.score_player0 - self.score_player1) >= 2:
            return True
        else:
            return False

    def _record_history(self, step: int, state: Tuple[int, int, int, int, int, int, int, int], action: Tuple[int, Tuple[int, int], int, str], next_state: Tuple[int, int, int, int, int, int, int, int], reward: int, failure_reason: str = "", losing_player: int = -1):
        """
        记录状态、动作、下一个状态和奖励的历史
        """
        self.history[step] = {
            'state': state,
            'action': action,
            'next_state': next_state,
            'reward': reward,
            'score_player0': self.score_player0,
            'score_player1': self.score_player1,
            'failure_reason': failure_reason,
            'losing_player': losing_player
        }



    def save_history_to_csv(self, filename: str):
        """
        将历史记录保存为 CSV 文件
        """
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            # 写入表头
            writer.writerow(['Step', 'Player_X', 'Player_Y', 'Opponent_X', 'Opponent_Y', 'Ball_X', 'Ball_Y', 'Ball_Height', 'Current_Player', 'Action_Type', 'Landing_X', 'Landing_Y', 'Action_Height', 'Next_Player_X', 'Next_Player_Y', 'Next_Opponent_X', 'Next_Opponent_Y', 'Next_Ball_X', 'Next_Ball_Y', 'Next_Ball_Height', 'Next_Current_Player', 'Reward', 'Score_Player0', 'Score_Player1', 'Failure_Reason', 'Losing_Player'])
            # 写入每一步的记录
            for step, record in self.history.items():
                state = record['state']
                action = record['action']
                next_state = record['next_state']
                reward = record['reward']
                score_player0 = record['score_player0']
                score_player1 = record['score_player1']
                failure_reason = record.get('failure_reason', "")
                losing_player = record.get('losing_player', -1)
                if action is not None:
                    action_type = self.ACTIONS[action[0]] if action[0] < len(self.ACTIONS) else '未知'
                    landing_x, landing_y = action[1]
                    action_height = action[2]
                else:
                    action_type = None
                    landing_x = None
                    landing_y = None
                    action_height = None
                writer.writerow([step, *state, action_type, landing_x, landing_y, action_height, *next_state, reward, score_player0, score_player1, failure_reason, losing_player])


    def animate_ball(self, history, end_reason):
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 绘制羽毛球场地
        # 场地外框

        court = patches.Rectangle((0, 0), 6.1, 13.4, facecolor='#006400', alpha=0.3)  # 浅绿色
    
        ax.add_patch(court)
        
        # 双打底线(外线)
        ax.plot([0, 6.1], [0, 0], 'black', linewidth=2)  # 下底线
        ax.plot([0, 6.1], [13.4, 13.4], 'black', linewidth=2)  # 上底线

        # 单打底线(内线)
        ax.plot([0, 6.1], [0.46, 0.46], 'black', linewidth=1)  # 下场区单打底线
        ax.plot([0, 6.1], [12.94, 12.94], 'black', linewidth=1)  # 上场区单打底线

        # 单打边线
        ax.plot([0.46, 0.46], [0, 13.4], 'black', linewidth=1)  # 左单打边线
        ax.plot([5.64, 5.64], [0, 13.4], 'black', linewidth=1)  # 右单打边线

        # 双打边线
        ax.plot([0, 0], [0, 13.4], 'black', linewidth=2)  # 左双打边线
        ax.plot([6.1, 6.1], [0, 13.4], 'black', linewidth=2)  # 右双打边线

        # 网线 (在场地正中间)
        ax.plot([0, 6.1], [6.7, 6.7], 'black', linewidth=2)

        # 发球线 (T字形)
        ax.plot([0, 6.1], [5.18, 5.18], 'black', linewidth=1)  # 下场区发球线
        ax.plot([0, 6.1], [8.22, 8.22], 'black', linewidth=1)  # 上场区发球线

        # 中间发球线
        ax.plot([3.05, 3.05], [0, 5.18], 'black', linewidth=1)  # 下半场
        ax.plot([3.05, 3.05], [8.22, 13.4], 'black', linewidth=1)  # 上半场

        # 添加场地底色
        court_color = patches.Rectangle((0, 0), 6.1, 13.4, facecolor='#D5E6F1', alpha=0.3)
        ax.add_patch(court_color)
        # 球员和球的初始化
        player = ax.add_patch(patches.Circle((0, 0), 0.3, color='blue', label='Player 0 (Blue)', visible=False))
        opponent = ax.add_patch(patches.Circle((0, 0), 0.3, color='red', label='Player 1 (Red)', visible=False))
        ball = ax.add_patch(patches.Circle((0, 0), 0.15, color='yellow', edgecolor='black', visible=False))
        
        # 信息显示面板
        info_box = patches.Rectangle((6.3, 5), 2, 7, facecolor='white', alpha=0.7)
        ax.add_patch(info_box)
        
        action_text = ax.text(6.5, 10, '', fontsize=10)
        reward_text = ax.text(6.5, 9, '', fontsize=10)
        player_text = ax.text(6.5, 8, '', fontsize=10)
        ball_height_text = ax.text(6.5, 7, '', fontsize=10)
        score_text = ax.text(6.5, 11, '', fontsize=10, weight='bold')
        failure_reason_text = ax.text(6.5, 6, '', fontsize=10, color='red',
            bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.5'),
            visible=False)

    
        def convert_coordinates(x, y, is_ball=False):
            """
            坐标转换函数
            is_ball: True表示是球的坐标转换,允许出界; False表示是球员的坐标转换,限制在场地内
            """
            # 基础转换
            court_x = 0.76 + y * ((5.34 - 0.76) / 2)
            court_y = 12.64 - x * ((12.64 - 0.76) / 5)

            if not is_ball:
                # 球员坐标限制在场地内
                court_x = max(0.76, min(5.34, court_x))
                court_y = max(0.76, min(12.64, court_y))
            else:
                # 球的坐标允许出界
                # 可以根据需要扩展显示范围
                court_x = max(-1, min(7.1, court_x))  # 允许球出界1米
                court_y = max(-1, min(14.4, court_y))  # 允许球出界1米

            return court_x, court_y


        def update(frame_data):
            step, t = frame_data
            
            if step >= len(history):
                return [ball, player, opponent, action_text, reward_text, 
                    player_text, ball_height_text, failure_reason_text, score_text]
                
            current_state = history[step]['state']
            next_state = history[step]['next_state']
            action = history[step]['action']
            reward = history[step]['reward']
            score_player0 = history[step]['score_player0']
            score_player1 = history[step]['score_player1']
            failure_reason = history[step].get('failure_reason', "")
            losing_player = history[step].get('losing_player', -1)

            # 更新位置信息显示
            if action is not None:
                action_idx = action[0]
                action_name = self.ACTIONS[action_idx] if action_idx < len(self.ACTIONS) else self.ACTIONS[-1]
                action_text.set_text(f'Action: {action_name}')
                reward_text.set_text(f'Reward: {reward}')
                player_text.set_text(f'Current Player: {current_state[-1]}')
                ball_height_text.set_text(f'Ball Height: {next_state[6]}')
                score_text.set_text(f'Score - Player 0: {score_player0}\nPlayer 1: {score_player1}')

            # 计算插值位置
            player_x = (1 - t) * current_state[0] + t * next_state[0]
            player_y = (1 - t) * current_state[1] + t * next_state[1]
            opponent_x = (1 - t) * current_state[2] + t * next_state[2]
            opponent_y = (1 - t) * current_state[3] + t * next_state[3]
            ball_x = (1 - t) * current_state[4] + t * next_state[4]
            ball_y = (1 - t) * current_state[5] + t * next_state[5]

            # 转换为场地坐标
            p_x, p_y = convert_coordinates(player_x, player_y, is_ball=False)
            o_x, o_y = convert_coordinates(opponent_x, opponent_y, is_ball=False)
            b_x, b_y = convert_coordinates(ball_x, ball_y, is_ball=True)

            # 更新位置
            player.set_visible(True)
            player.set_center((p_x, p_y))
            opponent.set_visible(True)
            opponent.set_center((o_x, o_y))
            ball.set_visible(True)
            ball.set_center((b_x, b_y))

            # 显示失误信息
            if failure_reason and t >= 1.0:
                failure_reason_text.set_text(f'Player {losing_player}\nlose: {failure_reason}')
                failure_reason_text.set_visible(True)
            else:
                failure_reason_text.set_visible(False)

            return [ball, player, opponent, action_text, reward_text, 
                    player_text, ball_height_text, failure_reason_text, score_text]

        # # 设置图形范围和属性
        # ax.set_xlim(-0.5, 8)
        # ax.set_ylim(-0.5, 14)
        # ax.set_aspect('equal')
        # ax.axis('off')
        # 设置图形范围和属性
        ax.set_xlim(-2, 8.5)
        ax.set_ylim(-2, 16)
        ax.set_aspect('equal')
        ax.axis('off')  
        # 添加图例和标注
        legend_elements = [
            patches.Patch(color='blue', label='Player 0 (Blue)'),
            patches.Patch(color='red', label='Player 1 (Red)'),
            # patches.Patch(facecolor='none', edgecolor='black', linewidth=2, label='Double Court'),
            # patches.Patch(facecolor='none', edgecolor='black', linewidth=1, label='Single Court')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.2, 1))


        # 创建动画
        fps = 10
        animate_frames_per_step = 10
        pause_frames = fps * 2
        
        total_frames = []
        for step in range(len(history)):
            for f in range(animate_frames_per_step):
                total_frames.append((step, f / animate_frames_per_step))
            
            failure_reason = history[step].get('failure_reason', "")
            if failure_reason and failure_reason != "success":
                for f in range(pause_frames):
                    total_frames.append((step, 1.0))
        
        ani = FuncAnimation(fig, update, frames=total_frames, blit=False, 
                        repeat=False, interval=1000/fps)
        
        plt.show()




    def render(self):
        """
        渲染击打过程
        """
        try:
            self.animate_ball(self.history, self.end_reason)
        except Exception as e:
            print(f"An error occurred during rendering: {e}")

    


