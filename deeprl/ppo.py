import numpy as np
import torch as th


def train_ppo(model, optimizer,
              obs, acs, advs, vtargs, old_ac_logps,
              n_epochs=3, n_mbatch=1, loss='clip',
              vfcoef=0.5, entcoef=0.01, kl_threshold=np.inf, **update_kwargs):
    
    batch_size = obs.shape[0]
    mbatch_size = int(batch_size / n_mbatch)
    for _ in range(n_epochs):
        shuffle_idxs = np.random.permutation(batch_size)
        obs, acs, advs, vtargs, old_ac_logps = map(
            lambda x: x[shuffle_idxs],
            [obs, acs, advs, vtargs, old_ac_logps])
        for i in range(n_mbatch):
            mb_obs, mb_acs, mb_advs, mb_vtargs, mb_old_ac_logps = map(
                lambda x: x[i * mbatch_size : (i+1) * mbatch_size],
                [obs, acs, advs, vtargs, old_ac_logps])

            if loss == 'clip':
                train_info = ppo_clip_update(model, optimizer,
                                             mb_obs, mb_acs, mb_advs, mb_vtargs, mb_old_ac_logps,
                                             vfcoef, entcoef, **update_kwargs)

            elif loss == 'klpen':
                raise NotImplementedError('PPO KL penalty loss not yet implemented')
            else:
                raise ValueError('loss must either be "clip" or "klpen"')

        if train_info['kl_divergence'] > kl_threshold:
            break

    return train_info


def ppo_clip_update(model, optimizer, obs, acs, advs, vtargs, old_ac_logps,
                    vfcoef=0.5, entcoef=0.01, clip_eps=0.2):

    pd, vpreds = model.forward(obs)
    ac_logps = pd.log_prob(acs).unsqueeze(-1)
    entropy = pd.entropy().unsqueeze(-1)
    reduce_acd_dims = [i for i in range(1, ac_logps.ndim)]
    ac_logps = ac_logps.sum(dim=reduce_acd_dims)
    entropy = entropy.sum(dim=reduce_acd_dims).mean()
    approx_kl = (old_ac_logps - ac_logps).mean()

    ac_logp_frac = th.exp(ac_logps - old_ac_logps)
    clipped_ac_logp_frac = th.clamp(ac_logp_frac, 1 - clip_eps, 1 + clip_eps)

    policy_loss = th.max(ac_logp_frac * -advs, clipped_ac_logp_frac * -advs).mean()
    value_loss = vfcoef * ((vpreds[:, 0] - vtargs) ** 2).mean()
    entropy_loss = -entropy * entcoef
    loss = policy_loss + value_loss + entropy_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_info_dict = dict(loss_policy=policy_loss,
                           loss_value=value_loss,
                           loss_entropy=entropy_loss,
                           entropy=entropy,
                           kl_divergence=approx_kl)

    return train_info_dict
