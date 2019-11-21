import torch as th


class RolloutGenerator(object):

    def __init__(self, model, env, gamma=0.99, lam=0.95):
        self.model = model
        self.env = env
        self.ob = env.reset()
        self.gamma = gamma
        self.lam = lam

        ac_d, vpred = self.model.forward(self.ob)
        ac = ac_d.sample()
        ac_logp = ac_d.log_prob(ac).unsqueeze(-1)
        self.reduce_acd_dims = [i for i in range(1, ac_logp.ndim)]

    def generate_rollout(self, rollout_len, normalize_advantage=True):
        obs = [self.ob]
        rews = []
        dones = []
        acs = []
        ac_logps = []
        vpreds = []
        epinfos = []

        with th.no_grad():
            for i in range(rollout_len):
                ac_d, vpred = self.model.forward(self.ob)
                ac = ac_d.sample()

                # we do the below summation so that we get the total logprob of the action, rather than
                # an array of the logprobs of subcomponents of the action
                # E.g. if the action is to apply force to several motors, this will give us the overall log
                # prob as opposed to an array of the log probs of the forces applied to each motor
                ac_logp = ac_d.log_prob(ac).unsqueeze(-1).sum(dim=self.reduce_acd_dims)

                self.ob, rew, done, infos = self.env.step(ac.numpy())

                obs.append(self.ob)
                rews.append(rew)
                dones.append(done)
                acs.append(ac)
                ac_logps.append(ac_logp)
                vpreds.append(vpred)
                for info in infos:
                    maybeepinfo = info.get('episode')
                    if maybeepinfo:
                        epinfos.append(maybeepinfo)

            _, last_vpred = self.model.forward(self.ob)
            vpreds.append(last_vpred)

        obs = th.as_tensor(obs, dtype=th.float)
        rews = th.as_tensor(rews, dtype=th.float)
        dones = th.as_tensor(dones, dtype=th.float)
        acs = th.stack(acs)
        ac_logps = th.stack(ac_logps)
        vpreds = th.stack(vpreds)
        advs = compute_gae(rews, vpreds[:, :, 0], dones, self.gamma, self.lam)
        vtargs = advs + vpreds[:-1, :, 0]

        obs, rews, dones, acs, ac_logps, vpreds, advs, vtargs = map(
            _sf01, (obs[:-1], rews, dones, acs, ac_logps, vpreds[:-1], advs, vtargs))

        if normalize_advantage:
            advs = (advs - advs.mean()) / advs.std()

        return obs, rews, dones, acs, ac_logps, vpreds, advs, vtargs, epinfos


def compute_gae(rews, vpreds, dones, gamma, lam):
    advs = th.zeros(vpreds.shape)
    for i in reversed(range(rews.shape[0])):
        advs[i] = ((-vpreds[i] + rews[i] + gamma * (1 - dones[i]) * vpreds[i+1]) +
                   gamma * lam * (1 - dones[i]) * advs[i+1])

    return advs[:-1]


def _sf01(tensor):
    s = tensor.shape
    return tensor.transpose(0, 1).reshape(s[0] * s[1], *s[2:])
