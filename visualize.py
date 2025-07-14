import pygame
import math

class Visualizer:
    @staticmethod
    def animate(history, ACTIONS):
        pygame.init()
        screen_width, screen_height = 1200, 800
        screen = pygame.display.set_mode((screen_width, screen_height))
        FONT = pygame.font.SysFont('simhei', 16)
        pygame.display.set_caption('Badminton Visualizer')
        clock = pygame.time.Clock()
        FPS = 10

        WHITE = (255, 255, 255)
        BLUE = (0, 0, 255)
        RED = (255, 0, 0)
        GREEN = (0, 156, 85)

        # 场地原始尺寸（单位：米）
        court_width = 6.1
        court_height = 13.4
        scale = 50
        display_width = court_width * scale
        display_height = court_height * scale
        offset_x = (screen_width - display_width) / 2
        offset_y = (screen_height - display_height) / 2

        # 坐标转换函数
        def court_to_screen(x, y):
            screen_x = offset_x + x * scale
            screen_y = offset_y + (court_height - y) * scale
            return (int(screen_x), int(screen_y))

        def convert_coordinates(x, y):
            court_x = 0.76 + y * ((5.34 - 0.76) / 2)
            court_y = 12.64 - x * ((12.64 - 0.76) / 5)
            return court_to_screen(court_x, court_y)

        # 加载并缩放球的PNG图片
        try:
            ball_image = pygame.image.load('ball.png').convert_alpha()
            ball_diameter = int(0.5 * scale)
            ball_image = pygame.transform.scale(ball_image, (ball_diameter, ball_diameter))
        except FileNotFoundError:
            ball_image = None

        # 计算角度
        angles = []
        for step in range(len(history)):
            current_state = history[step]['state']
            next_state = history[step]['next_state']

            current_sx, current_sy = convert_coordinates(current_state[4], current_state[5])
            next_sx, next_sy = convert_coordinates(next_state[4], next_state[5])

            dx = next_sx - current_sx
            dy = next_sy - current_sy

            if dx == 0 and dy == 0:
                angle_deg = 0
            else:
                angle_rad = math.atan2(-dy, dx)
                angle_deg = math.degrees(angle_rad)

            angles.append(angle_deg)

        # 生成帧列表
        total_frames = []
        animate_fps = 10
        pause_frames = FPS * 2

        for step in range(len(history)):
            for f in range(animate_fps):
                total_frames.append((step, f / animate_fps))
            failure_reason = history[step].get('failure_reason', '')
            if failure_reason and failure_reason != '成功':
                for f in range(pause_frames):
                    total_frames.append((step, 1.0))

        # 绘制场地函数
        def draw_court():
            screen.fill(GREEN)

            def draw_line(start, end, width=2):
                pygame.draw.line(screen, WHITE,
                                court_to_screen(*start),
                                court_to_screen(*end), width)

            def draw_dash(start, end, width=2, dash_length=10):
                s = court_to_screen(*start)
                e = court_to_screen(*end)
                dx = e[0] - s[0]
                dy = e[1] - s[1]
                distance = math.hypot(dx, dy)
                if distance == 0:
                    return
                ux = dx / distance
                uy = dy / distance
                current_pos = [float(s[0]), float(s[1])]
                left = distance
                dash_on = True
                while left > 0:
                    step = min(dash_length, left)
                    next_pos = [current_pos[0] + step * ux, current_pos[1] + step * uy]
                    if dash_on:
                        pygame.draw.line(screen, WHITE, tuple(map(int, current_pos)), tuple(map(int, next_pos)), width)
                    current_pos = next_pos
                    left -= step
                    dash_on = not dash_on

            # 绘制场地线条
            draw_line((0, 0), (6.1, 0), 2)
            draw_line((0, 13.4), (6.1, 13.4), 2)
            draw_line((0, 0.46), (6.1, 0.46), 1)
            draw_line((0, 12.94), (6.1, 12.94), 1)
            draw_line((0.46, 0), (0.46, 13.4), 1)
            draw_line((5.64, 0), (5.64, 13.4), 1)
            draw_line((0, 0), (0, 13.4), 2)
            draw_line((6.1, 0), (6.1, 13.4), 2)
            draw_dash((0, 6.7), (6.1, 6.7), 2)  # 球网
            draw_line((0, 5.18), (6.1, 5.18), 1)
            draw_line((0, 8.22), (6.1, 8.22), 1)
            draw_line((3.05, 0), (3.05, 5.18), 1)
            draw_line((3.05, 8.22), (3.05, 13.4), 1)

        # 动画主循环
        current_frame = 0
        running = True

        while running and current_frame < len(total_frames):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            step, t = total_frames[current_frame]
            current_frame += 1

            # 获取当前状态
            current_state = history[step]['state']
            next_state = history[step]['next_state']
            action = history[step]['action']
            reward = history[step]['reward']
            score_player0 = history[step]['score_player0']
            score_player1 = history[step]['score_player1']
            failure_reason = history[step].get('failure_reason', '')
            losing_player = history[step].get('losing_player', -1)
            t0 = t * 0.6

            player_x = (1 - t) * current_state[0] + t * next_state[0]
            player_y = (1 - t) * current_state[1] + t * next_state[1]
            opponent_x = (1 - t) * current_state[2] + t * next_state[2]
            opponent_y = (1 - t) * current_state[3] + t * next_state[3]
            if losing_player != -1:
                if (losing_player == 0) ^ (failure_reason == '击球落地'):
                    opponent_x = (1 - t0) * current_state[2] + t0 * next_state[4]
                    opponent_y = (1 - t0) * current_state[3] + t0 * next_state[5]
                else:
                    player_x = (1 - t0) * current_state[0] + t0 * next_state[4]
                    player_y = (1 - t0) * current_state[1] + t0 * next_state[5]
            ball_x = (1 - t) * current_state[4] + t * next_state[4]
            ball_y = (1 - t) * current_state[5] + t * next_state[5]

            # 坐标转换
            player_pos = convert_coordinates(player_x, player_y)
            opponent_pos = convert_coordinates(opponent_x, opponent_y)
            ball_pos = convert_coordinates(ball_x, ball_y)

            # 绘制场地
            draw_court()

            # 绘制球员
            pygame.draw.circle(screen, BLUE, player_pos, int(0.3 * scale))
            pygame.draw.circle(screen, RED, opponent_pos, int(0.3 * scale))

            # 绘制球
            if ball_image is not None:
                angle_deg = angles[step]
                rotated_image = pygame.transform.rotate(ball_image, angle_deg)
                image_rect = rotated_image.get_rect(center=ball_pos)
                screen.blit(rotated_image, image_rect.topleft)
            else:
                # 如果没有图片，使用圆形
                pygame.draw.circle(screen, (255, 255, 0), ball_pos, int(0.15 * scale))
                pygame.draw.circle(screen, WHITE, ball_pos, int(0.15 * scale), 1)

            # 渲染文本信息
            info_x = screen_width - 210
            info_y = 50

            if action is not None:
                action_idx = action[0]
                action_name = ACTIONS[action_idx] if action_idx < len(ACTIONS) else ACTIONS[-1]
                screen.blit(FONT.render(f'动作: {action_name}', True, WHITE), (info_x, info_y))
                screen.blit(FONT.render(f'奖励: {reward}', True, WHITE), (info_x, info_y + 20))
                screen.blit(FONT.render(f'当前player: {current_state[-1]}', True, WHITE), (info_x, info_y + 40))
                screen.blit(FONT.render(f'球高度: {next_state[6]:.2f}', True, WHITE), (info_x, info_y + 60))
                screen.blit(FONT.render(f'比分 - P0: {score_player0}  P1: {score_player1}', True, WHITE), (info_x, info_y + 80))

            # 失误信息
            if failure_reason and t >= 1.0:
                fail_text = FONT.render(f'Player {losing_player} {failure_reason}', True, RED)
                screen.blit(fail_text, (info_x, info_y + 120))

            # 图例
            pygame.draw.rect(screen, BLUE, (info_x, info_y + 160, 15, 15))
            screen.blit(FONT.render('Player 0', True, WHITE), (info_x + 20, info_y + 160))
            pygame.draw.rect(screen, RED, (info_x, info_y + 180, 15, 15))
            screen.blit(FONT.render('Player 1', True, WHITE), (info_x + 20, info_y + 180))

            pygame.display.flip()
            clock.tick(FPS)

        pygame.quit()