import numpy as np
import gym
import click
import time
from functools import partial
from collections import deque
import mujoco_py

from baselines import logger
from baselines.bench import Monitor
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

from torch import optim

from deeprl.models import mlp_model
from deeprl.rollouts import RolloutGenerator
from deeprl.a2c import train_a2c
from deeprl.ppo import train_ppo_clip


def make_monitored_env(env_name):
    env = gym.make(env_name)
    return Monitor(env, None, allow_early_resets=True)


@click.command()
@click.option('--env', '-e', type=str, default='CartPole-v0')
@click.option('--n-envs', type=int, default=4)
@click.option('--rollout-len', '-r', type=int, default=1000)
@click.option('--n-total-steps', '-ns', type=float, default=3e6)
@click.option('--log-interval', '-l', type=int, default=5)
@click.option('--algorithm', '-a', type=click.Choice(('a2c', 'ppo_clip')), default='a2c')
@click.option('--n-epochs', '-ne', type=int, default=10)
@click.option('--num-mbatch', '-mb', type=int, default=1)
def main(env, n_envs, rollout_len, n_total_steps, log_interval, algorithm, n_epochs, num_mbatch,
         entcoef=0, gamma=0.99, lam=0.97, kl_threshold=0.075):
    env = SubprocVecEnv([partial(make_monitored_env, env) for _ in range(n_envs)])

    model = mlp_model(env)
    rollout_generator = RolloutGenerator(model, env, gamma=gamma, lam=lam)
    optimizer = optim.Adam(model.parameters())

    n_batch = rollout_len * n_envs
    mbatch_size = int(n_batch / num_mbatch)
    epinfobuf = deque(maxlen=100)
    n_steps_per_second = deque(maxlen=log_interval)
    for update in range(1, int(n_total_steps / n_batch) + 1):
        update_start_time = time.time()
        obs, rews, dones, acs, old_ac_logps, vpreds, advs, vtargs, epinfos = (
            rollout_generator.generate_rollout(rollout_len))

        epinfobuf.extend(epinfos)

        if algorithm == 'a2c':
            train_info = train_a2c(model, optimizer, obs, acs, advs, vtargs)

        if algorithm == 'ppo_clip':
            for _ in range(n_epochs):
                for i in range(num_mbatch):
                    mb_obs = obs[i * mbatch_size: (i+1) * mbatch_size]
                    mb_acs = acs[i * mbatch_size: (i+1) * mbatch_size]
                    mb_advs = advs[i * mbatch_size: (i+1) * mbatch_size]
                    mb_vtargs = vtargs[i * mbatch_size: (i+1) * mbatch_size]
                    mb_old_ac_logps = old_ac_logps[i * mbatch_size: (i+1) * mbatch_size]
                    train_info = train_ppo_clip(model, optimizer, mb_obs, mb_acs, mb_advs,
                                                mb_vtargs, mb_old_ac_logps, entcoef=entcoef)

                if train_info['kl_divergence'] > kl_threshold:
                    break

        n_steps_per_second.append(n_batch / (time.time() - update_start_time))
        if update % log_interval == 0:
            train_info = dict([(k, v.item()) for k, v in train_info.items()])
            eprews = [epinfo['r'] for epinfo in epinfobuf]
            eplens = [epinfo['l'] for epinfo in epinfobuf]
            logger.logkv('n_steps_per_second', np.mean(n_steps_per_second))
            logger.logkv('total_steps', update * n_batch)
            if len(epinfobuf) > 0:
                logger.logkv('eprew_mean', np.mean(eprews))
                logger.logkv('eprew_std', np.std(eprews))
                logger.logkv('eprew_min', np.min(eprews))
                logger.logkv('eprew_max', np.max(eprews))
                logger.logkv('eplen_mean', np.mean(eplens))
                logger.logkv('eplen_std', np.std(eplens))
                logger.logkv('eplen_min', np.min(eplens))
                logger.logkv('eplen_max', np.max(eplens))

            logger.logkv('vpred_mean', vpreds.mean().item())
            logger.logkv('vpred_std', vpreds.std().item())
            logger.logkv('vpred_min', vpreds.min().item())
            logger.logkv('vpred_max', vpreds.max().item())
            logger.logkvs(train_info)
            logger.dumpkvs()


if __name__ == '__main__':
    main()
