import numpy as np
import random

class ReplayBuffer(object):
    def __init__(self, size):
        """创建优先重放缓冲区。

        Parameters
        ----------
        size: int
            要存储在缓冲区中的最大转换数。 
            当缓冲区溢出时，旧的内存将被丢弃。
        """
        self._storage = []
        self._maxsize = int(size)
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def clear(self):
        self._storage = []
        self._next_idx = 0

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def encode_sample_simple(self, idxes, n):
        obses_t, actions, rewards, obses_tp1, dones = [[] for _ in range(n)], \
                                                      [[] for _ in range(n)], \
                                                      [[] for _ in range(n)], \
                                                      [[] for _ in range(n)], \
                                                      [[] for _ in range(n)]
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            for j in range(n):
                obses_t[j].append(obs_t[j])
                actions[j].append(action[j])
                obses_tp1[j].append(obs_tp1[j])
                rewards[j].append(reward[j])
                dones[j].append(done[j])

        for j in range(n):
            obses_t[j] = np.array(obses_t[j])
            actions[j] = np.array(actions[j])
            obses_tp1[j] = np.array(obses_tp1[j])
            rewards[j] = np.array(rewards[j])
            dones[j] = np.array(dones[j])
        return obses_t, actions, rewards, obses_tp1, dones

    def make_index(self, batch_size):
        return [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]

    def make_latest_index(self, batch_size):
        idx = [(self._next_idx - 1 - i) % self._maxsize for i in range(batch_size)]
        np.random.shuffle(idx)
        return idx

    def sample_index(self, idxes):
        return self._encode_sample_simple(idxes)

    def sample(self, batch_size):
        """采样一批经验数据。

        参数
        ----------
        batch_size: int
            要采样的过渡数量。

        输出
        -------
        obs_batch: np.array
            观察值的批次
        act_batch: np.array
            在给定 obs_batch 的情况下执行的动作批次
        rew_batch: np.array
            作为执行 act_batch 的结果而收到的奖励
        next_obs_batch: np.array
            执行 act_batch 后看到的下一组观察值
        done_mask: np.array
            若执行 act_batch[i] 导致回合结束则 done_mask[i] = 1, 否则为 0。
        """
        if batch_size > 0:
            idxes = self.make_index(batch_size)
        else:
            idxes = range(0, len(self._storage))
        return self._encode_sample(idxes)

    def collect(self):
        return self.sample(-1)  # 采集所有可用的经验数据
