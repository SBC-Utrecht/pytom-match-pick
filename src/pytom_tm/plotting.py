import numpy as np
import traceback
from importlib.util import find_spec
import itertools
from scipy.optimize import curve_fit
from scipy.special import erf

if find_spec("matplotlib") is None or find_spec("seaborn") is None:
    raise RuntimeError(
        "ROC estimation can only be done when matplotlib and seaborn are installed."
    )
else:
    import matplotlib.pyplot as plt
    import seaborn as sns
sns.set(context="talk", style="ticks")


class ScoreHistogramFdr:
    def __init__(self):
        self.fig = plt.figure(figsize=(5 * 2, 5))
        self.hist_ax = self.fig.add_subplot(1, 2, 1)
        self.fdr_ax = self.fig.add_subplot(1, 2, 2)

    def draw_histogram(self, scores, nbins=30, return_bins=False):
        y, x_hist, _ = self.hist_ax.hist(
            scores, bins=nbins, histtype="step", color="grey"
        )
        self.hist_ax.set_xlabel(r"${LCC}_{max}$")
        self.hist_ax.set_xlim(x_hist[0], x_hist[-1])
        self.hist_ax.set_ylabel("Frequency")
        if return_bins:
            return y, x_hist

    def draw_bimodal(self, x, y1, y2, ymax=None):
        # plot bimodal model and the gaussian particle population
        self.hist_ax.plot(
            x, y1, lw=3.5, alpha=0.9, color="tab:blue"
        )  # , label='Bimodal model')
        self.hist_ax.plot(
            x, y2, lw=4, alpha=0.9, color="tab:orange"
        )  # , label='True positives')
        # population = params[2:5]
        if ymax is not None:
            self.hist_ax.set_ylim(0, ymax)
        # self.hist_ax.legend(loc='upper right')

    def draw_score_threshold(self, x, ymax):
        self.hist_ax.vlines(
            x, 0, ymax, linestyle="dashed", label=f"Cutoff: {x:.2f}", color="black"
        )
        self.hist_ax.legend(loc="upper right")

    def draw_fdr_recall(self, fdr, recall, optimal_id, ruc):
        self.fdr_ax.scatter(fdr, recall, facecolors="none", edgecolors="gray", s=25)
        # add optimal threshold in green
        self.fdr_ax.scatter(
            fdr[optimal_id],
            recall[optimal_id],
            s=25,
            color="black",
            label=f"RUC: {ruc:.2f}",
        )
        self.fdr_ax.plot([0, 1], [0, 1], ls="--", c=".3", lw=1)
        self.fdr_ax.set_xlabel("FDR")
        self.fdr_ax.set_ylabel("Recall")
        self.fdr_ax.set_xlim(0, 1)
        self.fdr_ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        self.fdr_ax.set_ylim(0, 1)
        self.fdr_ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        self.fdr_ax.legend(loc="lower right")

    def write(self, filename, quality=300, transparency=False, bbox="tight"):
        plt.tight_layout()
        plt.savefig(filename, dpi=quality, transparent=transparency, bbox_inches=bbox)

    def display(self):
        plt.tight_layout()
        plt.show()


def check_square_fdr(fdr, recall, epsilon=1e-3):
    """
    @param fdr: list of fdr values
    @type  fdr: L{list}
    @param recall: list of recall values
    @type  recall: L{list}
    @param epsilon: tolerance for closeness to 0 and 1
    @type  epsilon: L{float}

    @return: boolean of whether the FDR is almost square within tolerance
    @rtype:  L{bool}

    @author: Marten Chaillet
    """

    # fdr and recall should contain values very close to 0 and 1, respectively
    # if function is square
    union = [
        (f, r)
        for f, r in zip(fdr, recall)
        if ((np.abs(0.0 - f) < epsilon) and (np.abs(1.0 - r) < epsilon))
    ]
    return bool(union)


def distance_to_diag(fdr, recall):
    """
    @param fdr: list of fdr values
    @type  fdr: L{list}
    @param recall: list of recall values
    @type  recall: L{list}

    @return: list of distance of each fdr, recall combination to diagonal line
    @rtype:  L{list}

    @author: Marten Chaillet
    """
    # two point on the diagonal to find the distance to
    lp1, lp2 = (0, 0), (1, 1)
    # list to hold distances
    distance = []
    for f, r in zip(fdr, recall):
        d = np.abs(
            (lp2[0] - lp1[0]) * (lp1[1] - r) - (lp1[0] - f) * (lp2[1] - lp1[1])
        ) / np.sqrt((lp2[0] - lp1[0]) ** 2 + (lp2[1] - lp1[1]) ** 2)
        distance.append(d)
    return distance


def calculate_histogram(scores, num_steps):
    # construct x and y array according to the given peak index
    # preferably input is already sorted
    scores.sort()  # this is sorted from lowest to highest
    min = scores[0]
    max = scores[-1]

    step = (max - min) / num_steps
    x = []
    for i in range(num_steps):
        x.append(min + i * step)
    x.append(max)

    y = []
    for i in range(num_steps):
        lower = x[i]
        upper = x[i + 1]
        n = len([v for v in scores if lower <= v <= upper])
        y.append(n)

    return x, y


def evaluate_estimates(estimated_positions, ground_truth_positions, tolerance):
    """
    Estimated_positions numpy array, ground truth positions numpy array
    :param estimated_positions:
    :type estimated_positions:
    :param ground_truth_positions:
    :type ground_truth_positions:
    :param tolerance:
    :type tolerance:
    :return:
    :rtype:
    """
    from scipy.spatial.distance import cdist

    n_estimates = estimated_positions.shape[0]
    matrix = cdist(estimated_positions, ground_truth_positions, metric="euclidean")
    correct = [0] * n_estimates
    for i in range(n_estimates):
        if matrix[i].min() < tolerance:
            correct[i] = 1
    return correct


def fdr_recall(correct_particles, scores):
    assert all(i > j for i, j in itertools.pairwise(scores)), print(
        "Scores list should be decreasing."
    )

    n_true_positives = sum(correct_particles)
    true_positives, false_positives = 0, 0
    fdr, recall = [], []
    for correct, score in zip(correct_particles, scores):
        if correct:
            true_positives += 1
        else:
            false_positives += 1
        if n_true_positives == 0:
            recall.append(0)
        else:
            recall.append(true_positives / n_true_positives)
        fdr.append(false_positives / (true_positives + false_positives))
    return fdr, recall


def get_distance(line, point):
    a1, b1 = line
    x, y = point
    a2 = -(1 / a1)
    b2 = y - a2 * x

    x_int = (b2 - b1) / (a1 - a2)
    y_int = a2 * x_int + b2

    return np.sqrt((x_int - x) ** 2 + (y_int - y) ** 2)


def distance_to_random(fdr, recall):
    auc = [0] * len(fdr)
    for i in range(len(fdr)):
        d = get_distance((1, 0), (fdr[i], recall[i]))  # AUC should be 1 at most not 1/2
        if recall[i] > fdr[i]:
            auc[i] = d
        else:
            auc[i] = -d
    return max(auc), np.argmax(auc)


# ========== functions for fitting ==========
# define gaussian function with parameters to fit
def gauss(x, mu, sigma, amp):
    return amp * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


# integral of gaussian with certain sigma and A
def gauss_integral(sigma, amp):
    # mu does not influence the integral
    return amp * np.abs(sigma) * np.sqrt(2 * np.pi)


# define bimodal function of two gaussians to fit both populations
def bimodal(x, mu1, sigma1, amp1, mu2, sigma2, amp2):
    return gauss(x, mu1, sigma1, amp1) + gauss(x, mu2, sigma2, amp2)


def plist_quality_gaussian_fit(
    lcc_max_values,
    score_volume,
    particle_peak_index,
    force_peak=False,
    output_figure_name=None,
    crop_hist=False,
    num_bins=30,
    n_tomograms=1,
):
    # read out the scores
    correlation_scores = np.array(sorted(lcc_max_values, reverse=True))

    # draw the histogram
    plot = ScoreHistogramFdr()
    y, x_hist = plot.draw_histogram(
        correlation_scores, nbins=num_bins, return_bins=True
    )

    try:
        # ===== fit bimodal distribution =====
        # adjust x to center of each bin so len(x)==len(y)
        x = (x_hist[1:] + x_hist[:-1]) / 2
        hist_step = x_hist[1] - x_hist[0]

        # noise gaussian std
        # noise_sigma = np.sqrt((score_volume.std() ** 2) * n_tomograms)
        # if n_tomograms > 1 else score_volume.std()
        noise_sigma = score_volume.std()
        noise_mean = score_volume.mean()
        noise_size = score_volume.size * n_tomograms

        # noise gaussian A value
        noise_a = ((noise_size) / (noise_sigma * np.sqrt(2 * np.pi))) * hist_step

        # expected values
        # left gaussian expectation:
        #     score volume is skewed gaussian, fit only sigma with upper limit
        #     (skewed because it only contains highest score at each position)
        # right gaussian expectation:
        #     mu ~ x[half] and A ~ y[half]
        expected = (noise_sigma, x[particle_peak_index], 0.1, y[particle_peak_index])
        # force peak of particle population to be at peak index
        if force_peak:
            bounds = (
                [noise_sigma, x[particle_peak_index] - 0.01, 0, 0],
                [noise_sigma * 1.5, x[particle_peak_index] + 0.01, 0.1, y[1]],
            )
        else:
            bounds = (
                [noise_sigma, x[int(len(x) * 0.25)], 0, 0],
                [noise_sigma * 1.5, x[-1], 0.1, y[1]],
            )

        # TODO use lambda expression to fix mu_1 and sigma_1
        # params_names = ['mu_1', 'sigma_1', 'A_1', 'mu_2', 'sigma_2', 'A_2']
        params_names = ["sigma_1", "mu_2", "sigma_2", "A_2"]
        # skip first position as the noise peak there is likely incorrect
        params, cov = curve_fit(
            lambda x, p1, p2, p3, p4: bimodal(x, noise_mean, p1, noise_a, p2, p3, p4),
            x[1:],
            y[1:],
            p0=expected,
            bounds=bounds,
            maxfev=2000,
        )  # max iterations argument: maxfev=2000)
        # give sigma of fit for each parameter
        sigma = np.sqrt(np.diag(cov))

        # print information about fit of the model
        print("\nfit of the bimodal model:")
        print("\testimated\t\tsigma")
        for n, p, s in zip(params_names, params, sigma):
            print(f"{n}\t{p:.3f}\t\t{s:.3f}")
        print("\n")

        noise, population = ((noise_mean, params[0], noise_a), tuple(params[1:4]))
        y_bimodal = bimodal(x, *noise, *population)
        y_gauss = gauss(x, *population)

        if crop_hist:
            plot.draw_bimodal(x, y_bimodal, y_gauss, ymax=3 * population[2])
        else:
            plot.draw_bimodal(x, y_bimodal, y_gauss)

        # ===== Generate a ROC curve =====
        roc_steps = 50
        x_roc = np.flip(np.linspace(x[0], x[-1], roc_steps))

        # find ratio of hist step vs roc step
        roc_step = (x[-1] - x[0]) / roc_steps
        delta = (
            hist_step / roc_step
        )  # can be used to divide true/false positives by per roc step
        # variable for total number of tp and fp
        n_false_positives = 0.0
        # list for storing probability of true positives and false positives
        # for each cutoff
        recall = (
            []
        )  # recall = TP / (TP + FN); TP + FN is the full area under the Gaussian curve
        fdr = []  # false discovery rate = FP / (TP + FP); == 1 - TP / (TP + FP)

        # find integral of gaussian particle population;
        # NEED TO DIVIDE BY HISTOGRAM BIN STEP
        population_integral = gauss_integral(population[1], population[2]) / hist_step
        print(
            f" - estimation total number of true positives: {population_integral:.1f}"
        )

        # should use CDF (cumulative distribution function) of Gaussian,
        # gives probability from -infinity to x
        def cdf(x):
            return 0.5 * (1 + erf((x - population[0]) / (np.sqrt(2) * population[1])))

        def gauss_noise(x):
            return gauss(x, *noise)

        for x_i in x_roc:
            # calculate probability of true positives x_i
            # n_true_positives += gauss_pop(x_i) / delta
            n_true_positives = (1 - cdf(x_i)) * population_integral

            # determine false positives up to this point, could also use CDF
            n_false_positives += gauss_noise(x_i) / delta

            # add probability
            recall.append(n_true_positives / population_integral)
            fdr.append(n_false_positives / (n_true_positives + n_false_positives))

        # find best classifier by calculating the rectangle under the curve
        # for each roc point
        recall = np.array(recall)
        fdr = np.array(fdr)
        rectangles = recall * (1 - fdr)
        cutoff, ruc = rectangles.argmax(), rectangles.max()

        # plot the threshold on the distribution plot for visual inspection
        plot.draw_score_threshold(x_roc[cutoff], max(y))
        print(f" - optimal correlation coefficient threshold is {x_roc[cutoff]:.3f}")
        print(
            (
                " - this threshold approximately selects ",
                f"{(1 - cdf(x_roc[cutoff])) * population_integral:.1f} particles",
            )
        )

        # plot the fdr curve
        plot.draw_fdr_recall(fdr, recall, cutoff, ruc)
        print("Rectangle Under Curve (RUC): ", ruc)

    except (RuntimeError, ValueError):
        # runtime error is because the model could not be fit,
        # in that case print error and continue with execution
        traceback.print_exc()

    if output_figure_name is None:
        plot.display()
    else:
        if output_figure_name.suffix not in [".svg", ".png"]:
            output_figure_name = output_figure_name + ".png"
        plot.write(output_figure_name)
