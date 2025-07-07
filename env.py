import numpy as np
import os
from gym import spaces
from typing import Tuple, Dict, List
import csv
from player import Player
from visualize import Visualizer

class Env:

    def __init__(self):
        self.current_player = 0 # 当前击球球员

        self.is_serve_phase = True
        self.history = []
        self.state_history = [] #存储3个state状态
        self.ACTIONS = ['发球', '扣杀', '高远球', '网前球', '吊球', '平抽球', '挑球', '扑球', '挡网', '切球']


        self.end_reason = "success"

        self.is_serve_phase = True

        # 分数记录
        self.score_player0 = 0
        self.score_player1 = 0
        self.winning_score = 10  # 比赛结束的分数

    def reset(self):
        #重置发球
        self.is_serve_phase = True
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

        # 重置分数
        # self.scores = [0, 0]
        # self.score_player0 = 0
        # self.score_player1 = 0

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
        
        print(self.state)
        
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

    def _check_shot_failure(self, action):

        # return current_player.result(action)

        # """简化的击球失败检查（根据动作类型返回预设结果）"""
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

    def render(self):
        """
        渲染击打过程
        """
        # try:
        # visualizer = Visualizer(self.history, self.end_reason, self.ACTIONS)
        # visualizer.run()
        Visualizer.animate_ball(self.history, self.end_reason, self.ACTIONS)
        # except Exception as e:
        #     print(f"An error occurred during rendering: {e}")