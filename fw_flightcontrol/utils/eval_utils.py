import numpy as np
from enum import IntEnum
from fw_jsbgym.trim.trim_point import TrimPoint

class State(IntEnum):
    ROLL = 0
    PITCH = 1
    AIRSPEED = 2

class StateNoVa(IntEnum):
    ROLL = 0
    PITCH = 1

class RefSequence():
    def __init__(self, num_refs: int=5, min_step_bound: int=300, max_step_bound: int=500,
                 roll_bound: float=60.0, pitch_bound: float=30.0, airspeed_bound: float=10.0):
        self.num_refs = num_refs
        self.min_step_bound = min_step_bound
        self.max_step_bound = max_step_bound
        self.pitch_bound = pitch_bound
        self.roll_bound = roll_bound
        self.airspeed_bound = airspeed_bound
        self.ref_steps = None
        self.ref_cnts = None
        self.roll_ref = 0.0
        self.pitch_ref = 0.0 
        self.airspeed_ref = TrimPoint().Va_kph


    def sample_steps(self, offset: int=0) -> None:
        self.ref_steps = np.ones((self.num_refs, 3), dtype=np.int32)
        self.ref_cnts = np.zeros(3, dtype=np.int8)  # one counter for each state ref : roll, pitch, airspeed

        # Generate the remaining print steps
        for fcs in range(3):
            for i in range(1, self.num_refs):
                last_step = self.ref_steps[i-1, fcs]
                next_step = np.random.randint(last_step + self.min_step_bound, 
                                              min(last_step + self.max_step_bound, 2000), 
                                              dtype=np.int16)
                self.ref_steps[i, fcs] = next_step

        self.ref_steps += offset  # Convert the steps to integers


    def sample_refs(self, step: int, env_num: int=0):
        for state in State:
            if self.ref_cnts[state] < self.num_refs:
                if step % self.ref_steps[self.ref_cnts[state], state] == 0:
                    if state == State.ROLL:
                        self.roll_ref = np.deg2rad(np.random.uniform(-self.roll_bound, self.roll_bound))
                    elif state == State.PITCH:
                        self.pitch_ref = np.deg2rad(np.random.uniform(-self.pitch_bound, self.pitch_bound))
                    elif state == State.AIRSPEED:
                        self.airspeed_ref = np.random.uniform(TrimPoint().Va_kph - self.airspeed_bound, 
                                                              TrimPoint().Va_kph + self.airspeed_bound)
                    self.ref_cnts[state] += 1

        return self.roll_ref, self.pitch_ref, self.airspeed_ref


def main():
    np.random.seed(42)
    np.set_printoptions(suppress = True)
    ref_seq = RefSequence()
    ref_seq.sample_steps()
    total_steps = 50_000
    steps_per_episode = 2000
    n_episodes = total_steps // steps_per_episode
    step_seq_arr: np.ndarray = np.zeros((n_episodes, ref_seq.num_refs, 3), dtype=np.int32)
    ref_seq_arr: np.ndarray = np.zeros((total_steps+1, 3), dtype=np.float32)
    simple_easy_ref_seq_arr: np.ndarray = np.zeros((n_episodes, 2), dtype=np.float32)
    simple_medium_ref_seq_arr: np.ndarray = np.zeros((n_episodes, 2), dtype=np.float32)
    simple_hard_ref_seq_arr: np.ndarray = np.zeros((n_episodes, 2), dtype=np.float32)

    step_seq_arr[0] = ref_seq.ref_steps

    # save the reference steps and values to a file
    for i in range(total_steps):
        ref_seq.sample_refs(i)
        ref_seq_arr[i] = np.array([ref_seq.roll_ref, ref_seq.pitch_ref, ref_seq.airspeed_ref])
        if i % 2000 == 0:
            ref_seq.sample_steps(i)
            print(i // 2000)
            step_seq_arr[i // 2000] = ref_seq.ref_steps # save the reference steps

    for i in range(total_steps // 2000):
        # simple refs easy : roll [-30, 30], pitch [-20, 20]
        roll_ref = np.deg2rad(np.random.uniform(-30, 30))
        pitch_ref = np.deg2rad(np.random.uniform(-20, 20))
        simple_easy_ref_seq_arr[i] = np.array([roll_ref, pitch_ref])

        # simple refs medium : roll [-45, -30]U[30, 45], pitch [-25, -20]U[20, 25]
        roll_ref = np.deg2rad(np.random.uniform(45, 30))
        pitch_ref = np.deg2rad(np.random.uniform(25, 20))
        roll_sign = np.random.choice([-1, 1])
        roll_ref *= roll_sign
        pitch_sign = np.random.choice([-1, 1])
        pitch_ref *= pitch_sign
        simple_medium_ref_seq_arr[i] = np.array([roll_ref, pitch_ref])

        # simple refs hard : roll [-60, -45]U[45, 60], pitch [-30, -25]U[25, 30]
        roll_ref = np.deg2rad(np.random.uniform(60, 45))
        pitch_ref = np.deg2rad(np.random.uniform(30, 25))
        roll_sign = np.random.choice([-1, 1])
        roll_ref *= roll_sign
        pitch_sign = np.random.choice([-1, 1])
        pitch_ref *= pitch_sign
        simple_hard_ref_seq_arr[i] = np.array([roll_ref, pitch_ref])


    # print(step_seq_arr)
    # print(ref_seq_arr)
    ref_seq_arr[-1] = ref_seq_arr[-2]
    ref_seq_arr = ref_seq_arr[1:]
    step_seq_arr = step_seq_arr - 1

    np.save("eval/step_seq_arr.npy", step_seq_arr)
    np.save("eval/ref_seq_arr.npy", ref_seq_arr)
    np.save("eval/simple_easy_ref_seq_arr.npy", simple_easy_ref_seq_arr)
    np.save("eval/simple_medium_ref_seq_arr.npy", simple_medium_ref_seq_arr)
    np.save("eval/simple_hard_ref_seq_arr.npy", simple_hard_ref_seq_arr)

    # read the reference steps and values from a file
    step_seq_arr = np.load("eval/step_seq_arr.npy")
    ref_seq_arr = np.load("eval/ref_seq_arr.npy")
    simple_easy_ref_seq_arr = np.load("eval/simple_ref_seq_arr.npy")
    simple_medium_ref_seq_arr = np.load("eval/simple_ref_seq_arr.npy")
    simple_hard_ref_seq_arr = np.load("eval/simple_ref_seq_arr.npy")

    print(step_seq_arr)
    print(ref_seq_arr)
    print(simple_easy_ref_seq_arr)
    print(simple_medium_ref_seq_arr)
    print(simple_hard_ref_seq_arr)

if __name__ == "__main__":
    main()