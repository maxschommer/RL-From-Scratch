
import io
import pstats
import cProfile
import gym
import numpy as np
import matplotlib.pyplot as plt

from itertools import product
from typing import Callable, List, Any, Union, Iterable


# Mountain Car

# The last few exercises involved a discrete state space. There wasn't much
# of a reason to use function approximators other than to figure out how they
# work. Here I want to build a continuous state space RL agent. In order to
# focus on the RL aspect, I'm going to be using OpenAI Gym's game environments
# in order to attempt my solutions. I will be working in the MountainCar environment.

def zero() -> float:
    """Initialization function which returns 0.

    Returns:
        float: 0.0
    """
    return 0.0


def gaussian_random(mean: float, variance: float) -> float:
    """Initialization function which returns a gaussian random
    variable with a given mean and variance.

    Args:
        mean (float): Mean (center) of the distribution.
        variance (float): Variance (standard deviation) of the distribution.

    Returns:
        float: Sample of gaussian distribution
    """
    return np.random.normal(mean, variance)


class CoarseCodedApproximator(object):
    """A coarse coded generic linear feature approximator, which
    can be used as a value/action-value function.
    """

    def __init__(self):
        self.theta = []
        self.in_range: Iterable[Any] = []

        self._feature_vector_packed: Iterable[Any] = []

    def initialize_theta(self,
                         num_features: int,
                         init_function: Callable,
                         init_function_args: List[Any]) -> np.ndarray:
        """Initializes the feature vector theta using an intialization function.

        Args:
            num_features (int): The number of features (or length of theta)
            init_function (Callable): A function which will be called for each element of theta
                in order to initialize it.
            init_function_args (List[Any]): A list of arguments for the initialization function.
                If no arguments are required, use an empty list.

        Returns:
            np.ndarray: An array of float values.
        """
        res = []
        for _ in range(num_features):
            res.append(init_function(*init_function_args))
        return np.asarray(res)

    def build(self, state_spaces: List[Union[gym.spaces.Discrete, gym.spaces.Box]],
              num_features_per_dim: int = 20,
              overlap: float = 0.5,
              theta_init: Callable = zero,
              theta_init_args: List[Any] = []):
        """Builds the feature approximator, intializing the feature vector.

        Args:
            state_spaces (List[Union[gym.spaces.Discrete, gym.spaces.Box]]): A list of either
                discrete or finitely bounded continuous state spaces. If continuous, then the
                `num_features_per_dim` value will be used to generate that number of features.
                If discrete, then all values will be used to create features.
            num_features_per_dim (int, optional): A uniform range with num_features_per_dim
                elements between the bounds of each dimension will be the centers of the features.
                Defaults to 20.
            overlap (float, optional): The amount of 'overlap' of the features. More overlap means
                that more features will be associated with a given value, and create a smoother
                function approximator. Defaults to 0.5.
            theta_init (Callable, optional): The initialization function for the feature vector
                theta. Defaults to zero.
            theta_init_args (List[Any], optional): The arguments for the initialization function
                for the feature vector theta. If no arguments are required then leave as default.
                Defaults to [].
        """
        feature_ranges = []
        in_range = []
        num_features = 1
        for state_space in state_spaces:
            if isinstance(state_space, gym.spaces.Box):
                for low, high in zip(state_space.low, state_space.high):
                    feature_range, step = np.linspace(
                        low, high, num_features_per_dim, retstep=True)
                    feature_ranges.append(feature_range)
                    num_features *= len(feature_range)
                    in_range.append(step*overlap)

            if isinstance(state_space, gym.spaces.Discrete):
                feature_ranges.append(np.arange(state_space.n))
                num_features *= state_space.n
                in_range.append(None)
        self._feature_vector_packed = feature_ranges

        self.theta = self.initialize_theta(
            num_features, theta_init, theta_init_args)
        self.in_range = in_range

    def get_binary_idxs(self, state: np.ndarray) -> np.ndarray:
        """Gets the indices of the feature vector theta which are
        activated by the given state vector.

        Args:
            state (np.ndarray): The state to evaluate.

        Returns:
            np.ndarray: An array of indices of the feature vector.
        """
        binary_vecs = []
        feat_dims = []
        for packed_feature, read_val, dist in zip(self._feature_vector_packed, state, self.in_range):
            if dist is not None:
                binary_vec = np.abs(packed_feature - read_val) < dist
            else:
                binary_vec = read_val == packed_feature
            feat_dims.append(len(binary_vec))
            binary_vecs.append(binary_vec)

        idxs = []
        feat_lens = []
        for i, binary_vec in enumerate(binary_vecs):
            feat_lens.append(len(binary_vec))
            idxs.append(list(np.flatnonzero(binary_vec)
                             * np.product(feat_dims[i+1:])))
        feat_lens.reverse()
        if (len(list(product(*idxs))) == 0):
            return []
        feat_vecs = np.sum(np.asarray(list(product(*idxs)), dtype=int), axis=1)
        return feat_vecs

    def evaluate(self, state: np.ndarray) -> float:
        """Evaluates the function approximator at a given state.

        Args:
            state (np.ndarray): The state to evaluate the function approximator

        Returns:
            float: The evaluated function approximator value.
        """
        idxs = self.get_binary_idxs(state)
        return np.sum(self.theta[idxs])


def eval_func(func_to_eval, arg_vals) -> np.ndarray:
    dims = []
    for arg_val in arg_vals:
        dims.append(len(arg_val))
    all_args = list(product(*arg_vals))
    res = []
    for arg in all_args:
        res.append(func_to_eval(arg))

    return np.asarray(res).reshape(dims)


def plot_2d(X, Y, Z, labels={"xlabel": "x", "ylabel": "y", "zlabel": "z"}):
    plt.ioff()
    xv, yv = np.meshgrid(X, Y, sparse=False, indexing='ij')
    plt.figure()
    ax = plt.axes(projection='3d')
    if (len(Z.shape) == 3):
        for i in range(Z.shape[2]):
            ax.plot_wireframe(xv, yv, Z[:, :, i], color=np.random.rand(3,))
    elif (len(Z.shape) == 2):
        ax.plot_wireframe(xv, yv, Z, color='black')
    else:
        pass
    ax.set_title('wireframe')
    ax.set_xlabel(labels["xlabel"])
    ax.set_ylabel(labels["ylabel"])
    ax.set_zlabel(labels["zlabel"])
    plt.show()


def plot_Q(Q):
    evQ = eval_func(Q.evaluate, Q._feature_vector_packed)
    plot_2d(Q._feature_vector_packed[0],
            Q._feature_vector_packed[1], -np.max(evQ, axis=2),
            labels={"xlabel": "Position", "ylabel": "Velocity", "zlabel": "Reward"})


class SarsaLambda(object):
    def __init__(self, env, num_features_per_dim=20, overlap=1, lam=.9, gamma=1, alpha=0.01, finite_bound=1):

        self.state_space = self.bound_finite(
            env.observation_space, finite_bound)
        self.action_space = self.bound_finite(env.action_space, finite_bound)
        self.Q = CoarseCodedApproximator()

        self.Q.build([self.state_space, self.action_space],
                     num_features_per_dim=num_features_per_dim, overlap=overlap)
        # self.Q_eval = eval_func(self.Q.evaluate, self.Q._feature_vector_packed)
        self.lam = lam
        self.gamma = gamma
        self.alpha = alpha

    def bound_finite(self, space, bound):
        if isinstance(space, gym.spaces.Box):
            b_high = space.high
            b_high[b_high == np.inf] = bound
            b_low = space.low
            b_low[b_low == np.inf] = -bound
            return gym.spaces.Box(b_low, b_high)
        if isinstance(space, gym.spaces.Discrete):
            return space

    def act_epsilon_greedy(self, s, N, N0=100):
        N_val = 0
        for a in range(self.action_space.n):
            N_val += np.sum(N[self.Q.get_binary_idxs([*s, a])])

        epsilon = N0 / (N_val + N0)
        # epsilon = 0.05
        action_vals = []
        for i in range(self.action_space.n):
            action_vals.append(self.Q.evaluate(np.asarray([*s, i])))
        if np.random.uniform() < epsilon:
            return self.action_space.sample()
        else:
            return np.argmax(action_vals)

    def train(self, episodes, N0=100, liveplot=False, liveplot_freq=1):
        ts = []
        if liveplot:
            plt.ion()
            xv, yv = np.meshgrid(
                self.Q._feature_vector_packed[0],
                self.Q._feature_vector_packed[1], sparse=False, indexing='ij')
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            wf = ax.plot_wireframe(
                xv, yv, np.max(self.Q_eval, axis=2), color='black')
            ax.set_xlabel("Position")
            ax.set_ylabel("Velocity")
            ax.set_zlabel("Reward")
            plt.show()
            plt.pause(.001)
        N = np.zeros_like(self.Q.theta)
        for _ in range(episodes):
            if liveplot:
                rand_t_offset = np.random.randint(0, liveplot_freq)
            E = np.zeros_like(self.Q.theta)
            s = env.reset()
            a = self.act_epsilon_greedy(s, N, N0=N0)
            t = 0

            while True:
                N[self.Q.get_binary_idxs([*s, a])] += 1
                s_prime, r, done, _ = env.step(a)
                env.render()

                a_prime = self.act_epsilon_greedy(s_prime, N, N0=N0)
                delta = r + self.gamma * \
                    self.Q.evaluate([*s_prime, a_prime]) - \
                    self.Q.evaluate([*s, a])
                # print(r)
                idxs = self.Q.get_binary_idxs([*s, a])
                E[idxs] += 1
                E = self.gamma*self.lam*E
                self.Q.theta += self.alpha*delta*E
                if (liveplot and ((t+rand_t_offset) % liveplot_freq == 0)):
                    new_evals = []
                    idxs = np.flatnonzero(E)
                    for idx in idxs:
                        new_evals.append(self.Q.theta[idx])
                    flat_Q_eval = self.Q_eval.flatten()
                    flat_Q_eval[idxs] = new_evals
                    self.Q_eval = flat_Q_eval.reshape(self.Q_eval.shape)
                    wf.remove()
                    wf = ax.plot_wireframe(
                        xv, yv, np.max(self.Q_eval, axis=2), color='black')
                    plt.pause(.001)
                if done:
                    print(
                        "Episode finished after {} timesteps with reward {} and avg timesteps {}".format(t+1, r, np.mean(ts[-100:])))
                    ts.append(t + 1)
                    break
                else:
                    t += 1
                    s = s_prime
                    a = a_prime
        env.close()


env = gym.make("MountainCar-v0")

# print(env.observation_space.high)

sl = SarsaLambda(env, lam=.9, num_features_per_dim=50,
                 overlap=2.5, alpha=0.005, gamma=0.99, finite_bound=10)
sl.train(1000, 100)

# plot_Q(sl.Q)
