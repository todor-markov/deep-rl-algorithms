import torch as th


def train_ppo_clip(model, optimizer, obs, acs, advs, vtargs, old_ac_logps,
                   clip_eps=0.2, vfcoef=0.5, entcoef=0.01):

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
