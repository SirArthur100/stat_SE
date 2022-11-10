import matplotlib.pyplot as plt
import numpy as np
import scipy
import matplotlib.animation as animation

from matplotlib.ticker import FormatStrFormatter
from scipy.optimize import curve_fit


class Bernoulli_CLT:
    def __init__(
        self,
        x: np.ndarray,
        name: str,
        s: float = 1,
        loc: float = 0,
        scale: float = 1,
        frames: int = 5,
        sample_size: int = 5,
        interval: int = 100,
    ):
        self.x = x
        self.s = s
        self.loc = loc
        self.scale = scale
        self.frames = frames
        self.name = name
        self.sample_size = sample_size
        self.interval = interval
        self.lognorm = scipy.stats._continuous_distns.lognorm_gen(
            a=0.0, name="lognorm"
        )
        self.current_values = np.zeros(self.frames)

    def drawGif(self):

        fig, ax = plt.subplots(
            2, 1, sharex=True, sharey=True, figsize=(10, 10)
        )

        ax[1].set_xlim((10, 40))
        ax[1].set_ylim((0, 0.06))
        ax_sec = ax[1].twinx()
        ax_sec.set_ylim((0, 1))

        fig.text(
            0.03,
            0.5,
            "probability of given Binomial distribution",
            ha="center",
            va="center",
            rotation=90,
            size=20,
        )
        fig.text(0.6, 0.92, f"sample size = {self.sample_size}", size=20)

        def funcInit():
            ax[0].cla()
            ax[1].cla()
            ax_sec.cla()

        def funToAnimate(frame):

            random_selection = self.lognorm.rvs(
                self.s, loc=self.loc, scale=self.scale, size=self.sample_size
            )

            self.current_values[frame] = np.mean(random_selection)

            ax[0].cla()
            ax[1].cla()
            ax_sec.cla()

            for rand_num in random_selection:
                ax[0].axvline(rand_num, 0, 0.5, c="black", linewidth=0.5)
            ax[0].axvline(
                np.mean(random_selection), 0, 0.6, c="red", linewidth=4
            )

            ax[0].plot(
                self.x,
                self.lognorm.pdf(
                    self.x, s=self.s, scale=self.scale, loc=self.loc
                ),
                c="blue",
                linewidth=1,
            )

            ax[0].set_xlim((10, 40))
            ax[0].set_ylim((0, 0.06))
            ax[0].tick_params(labelsize=20)
            ax[1].tick_params(labelsize=20)
            ax_sec.tick_params(labelsize=20)
            ax[1].set_xlim((10, 40))
            ax[1].set_ylim((0, 0.06))
            ax_sec.yaxis.set_major_formatter(FormatStrFormatter("%0.2f"))
            vals = ax_sec.hist(
                self.current_values[: frame + 1],
                100,
                (10, 40),
                color="red",
                alpha=0.6,
                edgecolor="black",
                density=True,
            )

            ax[1].plot(
                self.x,
                self.lognorm.pdf(
                    self.x, s=self.s, scale=self.scale, loc=self.loc
                ),
                c="blue",
                linewidth=1,
            )

            # print(vals[0])

            ax[1].set_xlabel("outcome", size=20, labelpad=10)
            ax[1].set_ylabel(" ", size=20, labelpad=10)
            ax_sec.set_ylim(0, max(vals[0]) + 0.02)
            ax_sec.set_ylabel(
                "probability of sample averages", size=20, labelpad=7
            )
            fig.tight_layout()

            if frame % 50 == 0:
                print("Current frame: ", frame)

        anim = animation.FuncAnimation(
            fig,
            funToAnimate,
            init_func=funcInit,
            interval=self.interval,
            frames=self.frames,
            repeat=False,
        )

        anim.save(f"{self.name}.gif")


################################################################################


def generatorFunc(number_of_tests=5):

    x = np.linspace(0, 60, 600)
    probabilities = []

    for i in range(number_of_tests):
        bernoulli = Bernoulli_CLT(
            x,
            f"bernoulli{2**i*2}",
            s=1,
            scale=15,
            loc=0,
            frames=200,
            interval=100,
            sample_size=2**i * 2,
        )

        bernoulli.drawGif()
        probabilities.append(bernoulli.current_values)

    plt.figure()

    alphas = []

    for i in range(1, number_of_tests + 1):
        alphas.append(1 / i**2)

    alphas.reverse()

    for idx, seq in enumerate(probabilities):
        plt.hist(
            seq,
            40,
            (0, 60),
            density=True,
            alpha=alphas[idx],
            color="red",
            edgecolor="black",
        )

    plt.show()


def highresFunc():

    x = np.linspace(0, 60, 600)

    bernoulli = Bernoulli_CLT(
        x,
        f"bernoulli_highres",
        s=1,
        scale=15,
        loc=0,
        frames=1000,
        interval=10,
        sample_size=400,
    )

    bernoulli.drawGif()

    plt.figure()

    vals = plt.hist(
        bernoulli.current_values,
        100,
        (10, 40),
        density=True,
        alpha=0.6,
        color="red",
        edgecolor="black",
    )

    def gaus(X, a, X_mean, sigma):
        return a * np.exp(-((X - X_mean) ** 2) / (2 * sigma**2))

    opt_params, covariance = curve_fit(
        gaus,
        np.linspace(10, 40, 100),
        vals[0],
        bounds=((0.1, 10, 0), (1, 40, 5)),
    )

    plt.plot(
        np.linspace(10, 40, 100),
        gaus(
            np.linspace(10, 40, 100),
            opt_params[0],
            opt_params[1],
            opt_params[2],
        ),
        color="b",
        linewidth=5,
    )
