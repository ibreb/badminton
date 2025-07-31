# 状态和动作
ACTIONS = ['发球', '扣杀', '高远球', '网前吊球', '吊球', '平抽球', '挑球', '扑球', '挡网', '切球']
FEATURES = ['type', 'landing_area', 'ball_height', 'backhand', 'aroundhead']
FEATURE_SIZES = [10, 9, 3, 2, 2]

# 网络输入长度
OBS_DIM_RESULT_MODEL = 69
OBS_DIM_DEFENSE_MODEL = 78
OBS_DIM_ACT_MODEL = 78

# 历史步数
WINDOW_SIZE = 3

# 动作对应耗时
ACTION_TIMES = [0.7073511586452762, 0.5796270810210877, 1.1805558823529412, 0.8840086956521741, 0.860728144989339, 0.6727619961612284, 1.2004454277286134, 0.5675393258426966, 0.8883247232472324, 0.8164098360655737]


def load_config(config_data: dict):
    """加载配置并设置全局变量"""
    global ACTIONS, FEATURES, FEATURE_SIZES, OBS_DIM_RESULT_MODEL, OBS_DIM_DEFENSE_MODEL, OBS_DIM_ACT_MODEL, WINDOW_SIZE
    
    ACTIONS = config_data['actions']
    FEATURES = config_data['features']
    FEATURE_SIZES = config_data['feature_sizes']
    OBS_DIM_RESULT_MODEL = config_data['obs_dim']['result_model']
    OBS_DIM_DEFENSE_MODEL = config_data['obs_dim']['defense_model']
    OBS_DIM_ACT_MODEL = config_data['obs_dim']['act_model']
    WINDOW_SIZE = config_data['window_size']
