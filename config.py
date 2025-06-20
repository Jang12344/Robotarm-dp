#!/usr/bin/env python

import warnings
import torch as t

class DefaultConfig(object):
    # global parameter
    env = 'RLReachEnv'   # env name, need to be the same as envs/__init__.py
    """Choose from RLReachEnv / RLPushEnv / RLPickEnv / RLCamReachEnv"""
    algo = 'DADDPG_MLP'  # algo name, need to be the same as algo/__init__.py中的名字一致
    """Choose from DDPG_MLP / TD3_MLP / DADDPG_MLP / DATD3_MLP / DARC_MLP / DDPG_CNN / TD3_CNN / DADDPG_CNN / DATD3_CNN / DARC_CNN"""
    vis_name = 'Reach_DADDPG'  # visdom env
    vis_port = 8097      # visdom port
    jsonfile = "visdata/push/updata_TD3/TD3.json"     # json file dir
    csvname = "visdata/push/updata_TD3/updata_TD3_"   # data save dir

    # reach env parameter
    reach_ctr = 0.02     # to control the robot arm moving rate every step
    reach_dis = 0.01     # to control the target distance

    # train parameter
    use_gpu = False       # user GPU or not
    device = t.device('cuda') if use_gpu else t.device('cpu')
    random_seed = 0
    num_episodes = 500   # number of training episodes
    n_train = 40         # number of network updates per episodes
    minimal_episodes = 5  # Minimum number of start rounds for the experience replay buffer
    max_steps_one_episode = 500  # Maximum number of simulation steps per round

    # net parameter
    actor_lr = 1e-3      # actor net learning rate
    critic_lr = 1e-3     # critic net learning rate
    hidden_dim = 256     # mlp hidden size
    batch_size = 256     # batch size

    # public algo parameter
    sigma = 0.1          # Standard Deviation of Gaussian Noise
    tau = 0.005          # Target network soft update parameters
    gamma = 0.98         # discount
    buffer_size = 1000000   # buffer size

    # DQN algo only
    epsilon = 0.01
    target_update = 10

    # TD3, DATD3 algo only
    policy_noise = 0.2   # policy noise
    noise_clip = 0.5     # noise clip
    policy_freq = 3      # Delay update frequency

    # DARC algo only
    q_weight = 0.2
    regularization_weight = 0.005

    # HER algo only
    her_ratio = 0.8      # her rate per batch

    # Diffusion Policy algo only
    num_diffusion_steps = 100     # 训练时的扩散步数
    num_inference_steps = 50      # 推理时的去噪步数 
    beta_schedule = "squaredcos_cap_v2"  # 噪声调度器类型: "linear", "squaredcos_cap_v2"
    prediction_type = "epsilon"   # 预测类型: "epsilon"(噪声) 或 "sample"(干净动作)
    ema_decay = 0.995            # EMA衰减率
    clip_sample = True           # 是否裁剪样本
    network_type = "mlp"         # 网络类型: "mlp" 或 "unet"
    horizon_steps = 16           # 预测时域长度
    action_horizon = 8           # 实际执行动作数量
    diffusion_lr = 3e-4          # Diffusion Policy学习率

    def _parse(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        """
        # 更新配置参数
        for k, v in kwargs.items():
            if not hasattr(self, k):
                # 警告还是报错，取决于你个人的喜好
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        opt.device = t.device('cuda') if opt.use_gpu else t.device('cpu')

        # 打印配置信息
        print('-------------------------------------------------------------------')
        print('==> Printing user config..')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                km = '[' + str(k) + ']'
                print('{:<25}{:<20}'.format(str(km), str(getattr(self, k))))  # {:<30d}含义是 左对齐，且占用30个字符位
        print('-------------------------------------------------------------------')

    def _parsehelp(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        """
        # 更新配置参数
        for k, v in kwargs.items():
            if not hasattr(self, k):
                # 警告还是报错，取决于你个人的喜好
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        opt.device = t.device('cuda') if opt.use_gpu else t.device('cpu')

        # 打印配置信息
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                km = '[' + str(k) + ']'
                print('            {:<18}--- {:<20}'.format(str(km), str(getattr(self, k))))  # {:<30d}含义是 左对齐，且占用30个字符位


opt = DefaultConfig()