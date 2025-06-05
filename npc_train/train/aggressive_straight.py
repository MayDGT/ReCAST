import timeit

import gym
import gym_apollo # must
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy

gym.logger.set_level(40)
run = 1
start = timeit.default_timer()
model = SAC('MlpPolicy', 'Apollo-v0', learning_starts=10000, train_freq=(1, "step"), tensorboard_log="./sac_apollo_tensorboard", use_sde=False, device="cuda", verbose=1, seed=594371)
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path="checkpoints", name_prefix="angry_straight")
model.learn(total_timesteps=int(200000), log_interval=1, callback=checkpoint_callback, reward_type="STL", sem="sss")

print("learn over")
model.save("sac_apollo_angry_straight" + "_" + str(run))
model.save_replay_buffer("model_buffer" + "_" + str(run))
end = timeit.default_timer()
print("training_time is" + str(end - start))
text_file = open("hc.txt", "a")
text_file.write('Time for ' + str(run) + ' is : ' + str(end - start) + '\n')
text_file.close()

print("over")

