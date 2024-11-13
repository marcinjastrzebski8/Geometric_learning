import numpy as np
import pennylane as qml
from circuits import ansatz_dict, embedding_dict
from losses import fidelity_loss
from examples.utils import (
    HiggsDataset,
    ConfigReader,
    weight_init,
    Permute,
    setup_save_dir,
    expressivity,
    DynamicEntanglement,
)
from circuits import fidelity_circuit, Ansatz3
from new_train_qae import Trainer
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    accuracy_score,
    precision_score,
    f1_score,
)
from pennylane import numpy as qnp
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
from jax import numpy as jnp
from sklearn.utils.multiclass import type_of_target
import seaborn as sns
import pandas as pd
from optax import cosine_similarity
import jax

plt.rcParams.update({"text.usetex": True, "font.family": "Lucida Grande"})
plt.rcParams["figure.dpi"] = 150


class Tester(ABC):
    def __init__(self, model_fn, loss_fn, params):
        self.model_fn = model_fn
        self.loss_fn = loss_fn
        self.params = params

    @abstractmethod
    def eval_loss_fn(self, datapoint):
        raise NotImplementedError

    @abstractmethod
    def compute_losses(self, test_data, test_labels):
        raise NotImplementedError

    def tpr_fpr(self, test_data, test_labels):
        scores = []
        for i in range(test_data.shape[0]):
            l = self.eval_loss_fn([test_data[i, :]])
            scores.append(l)
        fpr, tpr, _ = roc_curve(test_labels, scores, drop_intermediate=False)
        auc_score = roc_auc_score(test_labels, scores)
        return fpr, tpr, auc_score

    def get_fpr_around_tpr_point(
        self, fpr: np.ndarray, tpr: np.ndarray, tpr_working_point: float = 0.8
    ):
        ind = np.array([])
        low_bound = tpr_working_point * 0.999
        up_bound = tpr_working_point * 1.001
        while len(ind) == 0:
            ind = np.where(np.logical_and(
                tpr >= low_bound, tpr <= up_bound))[0]
            low_bound *= 0.99  # open the window by 1%
            up_bound *= 1.01
        fpr_window_no_zeros = fpr[ind][fpr[ind] != 0]
        one_over_fpr_mean = np.mean(1.0 / fpr_window_no_zeros), np.std(
            1.0 / fpr_window_no_zeros
        )
        # print(
        #    f"\nTPR values around {tpr_working_point} window with lower bound {low_bound}"
        #    f" and upper bound: {up_bound}"
        # )
        # print(
        #    f"Corresponding mean 1/FPR value in that window: {one_over_fpr_mean[0]:.3f} ± "
        #    f"{one_over_fpr_mean[1]:.3f}"
        # )
        return one_over_fpr_mean, 1.0 / fpr_window_no_zeros

    def auc_threshold(self, test_data, test_labels):
        scores = []
        for i in range(test_data.shape[0]):
            l = self.eval_loss_fn([test_data[i, :]])
            scores.append(l)
        fpr, tpr, t = roc_curve(test_labels, scores)
        J = tpr - fpr
        ix = np.argmax(J)
        best_threshold = t[ix]
        return best_threshold

    def pr_threshold(self, test_data, test_labels):
        scores = np.zeros(test_data.shape[0])
        for i in range(test_data.shape[0]):
            l = self.eval_loss_fn([test_data[i, :]])
            scores[i] = l
        p, r, t = precision_recall_curve(test_labels, scores)
        f1score = (2 * p * r) / (p + r)
        ix = np.argmax(f1score)
        return t[ix]

    def precision_recall(self, test_data, test_labels):
        scores = np.zeros(test_data.shape[0])
        for i in range(test_data.shape[0]):
            l = self.eval_loss_fn([test_data[i, :]])
            scores[i] = l
        # precision, recall, t = precision_recall_curve(list(test_labels), list(scores))
        # score = precision_score(test_labels, test_data)
        p, r, t = precision_recall_curve(test_labels, scores)
        f1scores = (2 * p * r) / (p + r)
        ix = np.argmax(f1scores)
        f1score = f1scores[ix]
        return p, r, t, f1score

    def accuracy_score(self, test_data, test_labels, threshold):
        scores = []
        for i in range(test_data.shape[0]):
            l = self.eval_loss_fn([test_data[i, :]])
            if l < threshold:
                scores.append(0)
            if l >= threshold:
                scores.append(1)

        accuracy = accuracy_score(test_labels, scores)
        return accuracy

    def f1_score(self, test_data, test_labels):
        scores = []
        for i in range(test_data.shape[0]):
            l = self.eval_loss_fn([test_data[i, :]])
            scores.append(l)
        scores = np.array(scores) / max(scores)
        f1 = f1_score(test_labels, scores)

        return f1

    def plot_roc_curve(self, tpr, fpr):
        bg_rejection = 1 - fpr
        plt.plot(tpr, bg_rejection)
        plt.show()

    def confusion_matrix(self, test_data, test_labels, threshold):
        scores = []
        for i in range(test_data.shape[0]):
            l = self.eval_loss_fn([test_data[i, :]])
            if l < threshold:
                scores.append(0)
            if l >= threshold:
                scores.append(1)

        cm = confusion_matrix(test_labels, scores)
        return cm


class QuantumTester(Tester):
    def __init__(self, model_fn, loss_fn, params, properties):
        super().__init__(model_fn, loss_fn, params)
        self.properties = properties

    def compute_losses(self, test_data, test_labels):
        losses = []
        for i in range(test_data.shape[0]):
            l = self.eval_loss_fn([test_data[i, :]])
            losses.append(l)

        losses = np.array(losses)
        bg_labels_idx = np.where(test_labels == 0)[0]
        sg_labels_idx = np.where(test_labels == 1)[0]

        bg_losses = losses[bg_labels_idx]
        sig_losses = losses[sg_labels_idx]

        return np.array(bg_losses), np.array(sig_losses)

    def eval_loss_fn(self, datapoint):
        feat = qnp.array(datapoint)
        l = self.loss_fn(self.params, feat, self.model_fn, self.properties)
        l = np.array(l)
        return l

    def histogram(self, test_data, test_labels):
        bg_scores = []
        sg_scores = []
        bg_true = []
        sg_true = []
        for i in range(test_data.shape[0]):
            l = self.model_fn(self.params, test_data[i, :], self.properties)
            label = test_labels[i]
            if label == 0:
                bg_scores.append(l)
                bg_true.append(test_data[i, :])
            if label == 1:
                sg_scores.append(l)
                sg_true.append(test_data[i, :])
        return (
            np.array(bg_scores),
            np.array(sg_scores),
            np.array(bg_true),
            np.array(sg_true),
        )

    def statevector_histogram(self, test_data, test_labels):
        bg_vecs = []
        sg_vecs = []
        bg_true = []
        sg_true = []
        dev = qml.device("default.qubit", wires=self.properties["total_wires"])

        @qml.qnode(device=dev)
        def feature_vec(feature):
            self.properties["embedding_fn"](
                feature, self.properties["input_size"])
            return qml.state()

        @qml.qnode(device=dev)
        def recon_vec(feature, params):
            self.properties["embedding_fn"](
                feature, self.properties["input_size"])
            self.properties["ansatz_fn"](
                params,
                wires=range(self.properties["input_size"]),
                config=self.properties,
            )
            for i in range(self.properties["trash_size"]):
                qml.measure(
                    wires=self.properties["latent_size"] + i, reset=True)

            qml.adjoint(
                self.properties["ansatz_fn"](
                    params,
                    wires=range(self.properties["input_size"]),
                    config=self.properties,
                )
            )
            return qml.state()

        for i in range(test_data.shape[0]):
            init_vector = feature_vec(test_data[i, :])
            recon_vector = recon_vec(test_data[i, :], self.params)
            label = test_labels[i]
            if label == 0:
                bg_vecs.append(recon_vector)
                bg_true.append(init_vector)

            if label == 1:
                sg_vecs.append(recon_vector)
                sg_true.append(init_vector)
        return (
            np.array(sg_true),
            np.array(sg_vecs),
            np.array(bg_true),
            np.array(bg_vecs),
        )

    def get_reconstructions(self, test_data, test_labels, title, save_dir):
        bg_losses = []
        sig_losses = []
        for i in range(test_data.shape[0]):
            l = self.eval_loss_fn([test_data[i, :]])
            l = np.array(l)
            if test_labels[i] == 0:
                bg_losses.append(l)
            if test_labels[i] == 1:
                sig_losses.append(l)

        bg_mean = np.round(np.mean(bg_losses), 3)
        bg_std = np.round(np.std(bg_losses), 4)
        bg_median = np.round(np.median(bg_losses), 3)

        sg_mean = np.round(np.mean(sig_losses), 3)
        sg_std = np.round(np.std(sig_losses), 4)
        sg_median = np.round(np.median(sig_losses), 3)

        plt.hist(
            bg_losses,
            bins=100,
            label=f"bg mean: {bg_mean}+/-{bg_std}, median: {bg_median}",
            alpha=0.5,
        )
        plt.hist(
            sig_losses,
            bins=100,
            label=f"sig mean: {sg_mean}+/-{sg_std}, median: {sg_median}",
            alpha=0.5,
        )
        plt.legend()
        plt.title(title)
        plt.savefig(save_dir)
        plt.close()


class ClassicalTester(Tester):
    def __init__(self, model_fn, loss_fn, params):
        super().__init__(model_fn, loss_fn, params)

    def compute_losses(self, test_data, test_labels):
        params = self.params
        model_fn = self.model_fn
        loss_fn = self.loss_fn

        @jax.jit
        def eval_step(datapoint):
            return loss_fn(params, datapoint, model_fn)

        losses = []
        for _, datapoint in enumerate(test_data):
            loss = eval_step(datapoint)
            losses.append(loss)

        bg_labels_idx = np.where(test_labels == 0)[0]
        sg_labels_idx = np.where(test_labels == 1)[0]

        losses = np.array(losses)
        bg_losses = losses[bg_labels_idx]
        sig_losses = losses[sg_labels_idx]

        return np.array(bg_losses), np.array(sig_losses)

    def eval_loss_fn(self, datapoint):
        datapoint = jnp.array(datapoint)

        @jax.jit
        def eval(datapoint, params):
            l = self.loss_fn(params, datapoint, self.model_fn)
            # l = jnp.array(l)
            return l

        l = eval(datapoint, self.params)
        return l

    def histogram(self, test_data, test_labels):
        bg_scores = []
        sg_scores = []
        bg_true = []
        sg_true = []
        for i in range(test_data.shape[0]):
            l = self.model_fn.apply(self.params, test_data[i, :])
            label = test_labels[i]
            if label == 0:
                bg_scores.append(l)
                bg_true.append(test_data[i, :])
            if label == 1:
                sg_scores.append(l)
                sg_true.append(test_data[i, :])
        return (
            np.array(bg_scores),
            np.array(sg_scores),
            np.array(bg_true),
            np.array(sg_true),
        )

    def get_reconstructions(self, test_data, test_labels, title, save_dir):
        bg_recons = []
        sig_recons = []
        bg_true = []
        sig_true = []

        for i in range(test_data.shape[0]):
            l = self.model_fn.apply(self.params, test_data[i, :])
            l = np.array(l)
            if test_labels[i] == 0:
                bg_recons.append(l)
                bg_true.append(test_data[i, :])
            if test_labels[i] == 1:
                sig_recons.append(l)
                sig_true.append(test_data[i, :])

        bg_recons = np.array(bg_recons)
        sig_recons = np.array(sig_recons)
        bg_true = np.array(bg_true)
        sig_true = np.array(sig_true)

        bg_similarities = 1 - cosine_similarity(bg_recons, bg_true)
        sg_similarities = 1 - cosine_similarity(sig_recons, sig_true)

        bg_mean = np.round(np.mean(bg_similarities), 3)
        bg_std = np.round(np.std(bg_similarities), 4)
        bg_median = np.round(np.median(bg_similarities), 3)

        sg_mean = np.round(np.mean(sg_similarities), 3)
        sg_std = np.round(np.std(sg_similarities), 4)
        sg_median = np.round(np.median(sg_similarities), 3)

        plt.hist(
            bg_similarities,
            bins=100,
            label=f"bg mean: {bg_mean}+/-{bg_std}, median: {bg_median}",
            alpha=0.5,
        )
        plt.hist(
            sg_similarities,
            bins=100,
            label=f"sig mean: {sg_mean}+/-{sg_std}, median: {sg_median}",
            alpha=0.5,
        )
        plt.legend()
        plt.title(title)
        plt.savefig(save_dir)
        plt.close()


def main():
    k_folds = 3

    config = ConfigReader().compile_config()

    config["embedding_fn"] = embedding_dict[config["embedding"]]
    config["ansatz_fn"] = Ansatz3
    dev = qml.device("default.qubit", wires=config["total_wires"])
    model_fn = qml.QNode(fidelity_circuit, dev)
    loss_fn = fidelity_loss

    # Trainer takes argument on how weights should be reset on each fold
    trainer = Trainer(k_folds=k_folds)

    optimiser = qml.AdamOptimizer(config["lr"])

    param_shape = list(Ansatz3.shape(config["input_size"], 1))

    train_data = HiggsDataset(
        "data/higgs_dataset",
        size=config["train_size"],
        partition="train",
        transform=Permute([0, 1, 2, 3]),
    )

    test_data = HiggsDataset(
        "data/higgs_dataset",
        size=5,
        partition="test",
        transform=Permute([0, 1, 2, 3]),
    )
    test_data, test_labels = test_data.get_test_chunk(200, 200)

    param_shape = Ansatz3.shape(config["input_size"], 1)
    init_params = weight_init(
        config["w_init_range"][0],
        config["w_init_range"][1],
        config["w_init_dist"],
        param_shape,
    )
    save_dir = setup_save_dir(config)

    params, loss_hist, info = trainer.train(
        train_data,
        config["train_size"],
        model_fn,
        loss_fn,
        optimiser,
        config["epochs"],
        config["batch_size"],
        init_params,
        circuit_properties=config,
        eval_interval=10,
        save_dir=save_dir,
        callbacks=[DynamicEntanglement],
    )
    x = np.linspace(0, config["epochs"], int(config["epochs"] / 10 + 1))
    loss_mean = np.mean(loss_hist, axis=0)
    loss_std = np.std(loss_hist, axis=0)

    plt.plot(x, loss_mean)
    plt.fill_between(x, loss_mean + loss_std, loss_mean - loss_std, alpha=0.5)

    plt.show()

    tester = Tester(model_fn, loss_fn, params, config)
    tpr, fpr = tester.tpr_fpr(test_data=test_data, test_labels=test_labels)
    tester.plot_roc_curve(tpr, fpr)

    plt.plot(info["dynamic_entanglement"], "x-")
    plt.show()


def heatmap(x, y, **kwargs):
    if "color" in kwargs:
        color = kwargs["color"]
    else:
        color = [1] * len(x)

    if "palette" in kwargs:
        palette = kwargs["palette"]
        n_colors = len(palette)
    else:
        n_colors = 256  # Use 256 colors for the diverging color palette
        palette = sns.color_palette("Blues", n_colors)

    if "color_range" in kwargs:
        color_min, color_max = kwargs["color_range"]
    else:
        color_min, color_max = min(color), max(
            color
        )  # Range of values that will be mapped to the palette, i.e. min and max possible correlation

    def value_to_color(val):
        if color_min == color_max:
            return palette[-1]
        else:
            val_position = float((val - color_min)) / (
                color_max - color_min
            )  # position of value in the input range, relative to the length of the input range
            val_position = min(
                max(val_position, 0), 1
            )  # bound the position betwen 0 and 1
            ind = int(
                val_position * (n_colors - 1)
            )  # target index in the color palette
            return palette[ind]

    if "size" in kwargs:
        size = kwargs["size"]
    else:
        size = [1] * len(x)

    if "size_range" in kwargs:
        size_min, size_max = kwargs["size_range"][0], kwargs["size_range"][1]
    else:
        size_min, size_max = min(size), max(size)

    size_scale = kwargs.get("size_scale", 500)

    def value_to_size(val):
        if size_min == size_max:
            return 1 * size_scale
        else:
            val_position = (val - size_min) * 0.99 / (
                size_max - size_min
            ) + 0.01  # position of value in the input range, relative to the length of the input range
            val_position = min(
                max(val_position, 0), 1
            )  # bound the position betwen 0 and 1
            return val_position * size_scale

    if "x_order" in kwargs:
        x_names = [t for t in kwargs["x_order"]]
    else:
        x_names = [t for t in sorted(set([v for v in x]))]
    x_to_num = {p[1]: p[0] for p in enumerate(x_names)}

    if "y_order" in kwargs:
        y_names = [t for t in kwargs["y_order"]]
    else:
        y_names = [t for t in sorted(set([v for v in y]))]
    y_to_num = {p[1]: p[0] for p in enumerate(y_names)}

    plot_grid = plt.GridSpec(
        1, 15, hspace=0.2, wspace=0.1)  # Setup a 1x10 grid
    ax = plt.subplot(
        plot_grid[:, :-1]
    )  # Use the left 14/15ths of the grid for the main plot

    marker = kwargs.get("marker", "s")

    kwargs_pass_on = {
        k: v
        for k, v in kwargs.items()
        if k
        not in [
            "color",
            "palette",
            "color_range",
            "size",
            "size_range",
            "size_scale",
            "marker",
            "x_order",
            "y_order",
            "xlabel",
            "ylabel",
        ]
    }

    ax.scatter(
        x=[x_to_num[v] for v in x],
        y=[y_to_num[v] for v in y],
        marker=marker,
        s=[value_to_size(v) for v in size],
        c=[value_to_color(v) for v in color],
        **kwargs_pass_on,
    )
    ax.set_xticks([v for k, v in x_to_num.items()])
    ax.set_xticklabels([k for k in x_to_num], rotation=45,
                       horizontalalignment="right")
    ax.set_yticks([v for k, v in y_to_num.items()])
    ax.set_yticklabels([k for k in y_to_num])

    ax.grid(False, "major")
    ax.grid(True, "minor")
    ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
    ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)

    ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5])
    ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])
    ax.set_facecolor("#F1F1F1")

    ax.set_xlabel(kwargs.get("xlabel", ""))
    ax.set_ylabel(kwargs.get("ylabel", ""))

    # Add color legend on the right side of the plot
    if color_min < color_max:
        # Use the rightmost column of the plot
        ax = plt.subplot(plot_grid[:, -1])

        col_x = [0] * len(palette)  # Fixed x coordinate for the bars
        bar_y = np.linspace(
            color_min, color_max, n_colors
        )  # y coordinates for each of the n_colors bars

        bar_height = bar_y[1] - bar_y[0]
        ax.barh(
            y=bar_y,
            width=[5] * len(palette),  # Make bars 5 units wide
            left=col_x,  # Make bars start at 0
            height=bar_height,
            color=palette,
            linewidth=0,
        )
        ax.set_xlim(
            1, 2
        )  # Bars are going from 0 to 5, so lets crop the plot somewhere in the middle
        ax.grid(False)  # Hide grid
        ax.set_facecolor("white")  # Make background white
        ax.set_xticks([])  # Remove horizontal ticks
        ax.set_yticks(
            np.linspace(min(bar_y), max(bar_y), 3)
        )  # Show vertical ticks for min, middle and max
        ax.yaxis.tick_right()  # Show vertical ticks on the right
    plt.savefig("confusion.pdf")
    plt.close()


def corrplot(data, size_scale=500, marker="s"):
    x = [0, 0, 1, 1]
    y = [0, 1, 0, 1]

    heatmap(
        x,
        y,
        color=data,
        color_range=[-1, 1],
        palette=sns.diverging_palette(20, 220, n=256),
        size=data,
        size_range=[0, 1],
        marker=marker,
        size_scale=size_scale,
    )


def make_confusion_matrix(
    cf,
    save_dir,
    group_names=None,
    categories="auto",
    count=True,
    percent=True,
    cbar=True,
    xyticks=True,
    xyplotlabels=True,
    sum_stats=True,
    figsize=None,
    cmap="Blues",
    title=None,
):
    """
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.

    Arguments
    ---------
    cf:            confusion matrix to be passed in

    group_names:   List of strings that represent the labels row by row to be shown in each square.

    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'

    count:         If True, show the raw number in the confusion matrix. Default is True.

    normalize:     If True, show the proportions for each category. Default is True.

    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.

    xyticks:       If True, show x and y ticks. Default is True.

    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.

    sum_stats:     If True, display summary statistics below the figure. Default is True.

    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.

    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html

    title:         Title for the heatmap. Default is None.

    """

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ["" for i in range(cf.size)]

    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = [
            "{0:.2%}%".format(value) for value in cf.flatten() / np.sum(cf)
        ]
    else:
        group_percentages = blanks

    box_labels = [
        f"{v1}{v2}{v3}".strip()
        for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)
    ]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(cf) / float(np.sum(cf))

        # if it is a binary confusion matrix, show some more stats
        if len(cf) == 2:
            # Metrics for Binary Confusion Matrices
            precision = cf[1, 1] / sum(cf[:, 1])
            recall = cf[1, 1] / sum(cf[1, :])
            f1_score = 2 * precision * recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy, precision, recall, f1_score
            )
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize == None:
        # Get default figure size if not set
        figsize = plt.rcParams.get("figure.figsize")

    if xyticks == False:
        # Do not show categories if xyticks is False
        categories = False

    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(
        cf,
        annot=box_labels,
        fmt="",
        cmap=cmap,
        cbar=cbar,
        xticklabels=categories,
        yticklabels=categories,
    )

    if xyplotlabels:
        plt.ylabel("True label")
        plt.xlabel("Predicted label" + stats_text)
    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(save_dir)
    plt.close()


if __name__ == "__main__":
    # main()
    d = np.array([[200, 20], [30, 400]])
    len(d)
    labels = ["True Negative", "False Positive",
              "False Negative", "True Positive"]
    categories = ["Background", "Signal"]
    make_confusion_matrix(
        d,
        categories=categories,
        group_names=labels,
        title="confusion matrix",
        save_dir="newnew.pdf",
    )
