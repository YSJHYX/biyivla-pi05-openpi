import threading

import numpy as np


class ActionQueue:
    def __init__(self, actions):
        # TODO:将 actions 转为 NumPy 数组,并复制保存
        self._actions_ = np.asarray(actions, dtype=np.float32).copy()
        # TODO:初始化下一行动作的索引为 0
        self._next_index_ = 0
        # TODO:初始化总控制周期 tick 为 0
        self._tick_ = 0
        # TODO:创建 threading.Lock,供下面两个方法共用
        self._lock_ = threading.Lock()

    def get(self):
        # TODO:进入锁保护区域
        with self._lock_:
            # TODO:如果还有动作,取出当前行的副本,并将索引增加 1
            if self._next_index_ < len(self._actions_):
                action = self._actions_[self._next_index_].copy()
                self._next_index_ += 1
            # TODO:如果队列已耗尽,准备返回 None
            else:
                action = None
            # TODO:无论是否取到动作,都将 tick 增加 1
            self._tick_ += 1
            # TODO:返回动作副本或 None
        return action

    def snapshot(self):
        # TODO:进入锁保护区域
        with self._lock_:
            # TODO:读取当前 tick
            tick = self._tick_
            # TODO:从当前索引开始,复制所有尚未取出的动作
            remaining_actions = self._actions_[self._next_index_ :].copy()
            # TODO:返回 (tick, 剩余动作副本)
            # TODO:保持索引和 tick 不变
        return (tick, remaining_actions)

    def merge(self, actions, request_tick):
        # TODO:将新 actions 转为 float32 NumPy 数组,并复制保存
        new_actions = np.asarray(actions, dtype=np.float32).copy()
        # TODO:进入同一把锁的保护区域
        with self._lock_:
            # TODO:计算实际延迟:当前 tick - request_tick
            actual_delay = self._tick_ - request_tick
            # TODO:如果实际延迟小于 0,抛出 ValueError
            # 提示:请求时间不能来自未来
            if actual_delay < 0:
                raise ValueError("Request tick cannot be from the future.")
            # TODO:如果实际延迟已达到或超过新动作长度,
            #       返回 False,保持原队列、索引和 tick 不变
            if actual_delay >= len(new_actions):
                return False
            # TODO:丢弃新动作开头已经过期的行,
            #       用剩余新动作替换当前动作数组
            self._actions_ = new_actions[actual_delay:].copy()
            # TODO:将下一行动作的索引重置为 0
            self._next_index_ = 0
        # TODO:保持总 tick 不变,返回 True
        return True
