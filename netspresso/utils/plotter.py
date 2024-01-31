import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    BAR_WIDTH = 0.45

    @staticmethod
    def _add_text_annotation(ax, x_point, y_point, text):
        ax.annotate(
            f"{text:.2f}",
            (x_point, y_point),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            va="bottom",
        )

    @staticmethod
    def _add_value_annotations(ax, bar, value):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{value:.4f}",
            ha="center",
            va="bottom",
        )

    @staticmethod
    def _add_difference_annotations(ax, original_value, compressed_value, difference):
        ax.scatter(
            ["Original Model", "Compressed Model"],
            [original_value, compressed_value],
            color="red",
            marker="o",
            zorder=3,
        )

        ax.plot(
            ["Original Model", "Compressed Model"],
            [original_value, compressed_value],
            color="red",
            linestyle="--",
            linewidth=2,
            zorder=2,
        )

        diff_x = np.mean(ax.get_xlim())
        ax.text(
            diff_x,
            original_value * 0.5,
            f"Difference: {difference:.4f}",
            ha="center",
            va="bottom",
            color="black",
        )

    @staticmethod
    def _plot_single_bar(ax, label, value, color):
        return ax.bar(
            [label],
            [value],
            color=color,
            label=label,
            width=Plotter.BAR_WIDTH,
        )

    @staticmethod
    def _set_common_plot_settings(ax, ylabel, ylim_max=1):
        ax.set_ylim(0, ylim_max)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(axis="y")

    @staticmethod
    def _plot_comparison(original_value, value_per_model, target_value, title, xlabel, ylabel):
        ratios = list(value_per_model.keys())
        values = list(value_per_model.values())
        plt.figure(figsize=(15, 6))
        plt.plot(ratios, values, marker="o", label="Compressed Model")
        plt.axhline(original_value, color="slategray", linestyle="--", label="Original Model")
        if target_value:
            plt.axhline(target_value, color="red", linestyle="--", label="Target Value")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)

        for ratio, value in value_per_model.items():
            Plotter._add_text_annotation(plt.gca(), ratio, value, value)

    @staticmethod
    def compare_latency(original_latency, latency_per_model, target_latency):
        Plotter._plot_comparison(
            original_latency,
            latency_per_model,
            target_latency,
            "Latency vs. Compression Ratio",
            "Compression Ratio",
            "Latency (ms)",
        )

    @staticmethod
    def compare_flops(original_flops, flops_per_model):
        Plotter._plot_comparison(
            original_flops,
            flops_per_model,
            None,
            "FLOPs vs. Compression Ratio",
            "Compression Ratio",
            "FLOPs (M)",
        )

    @staticmethod
    def compare_metric(original_summary, compressed_summary):
        original_training_result = original_summary["traning_result"]
        compressed_training_result = compressed_summary["traning_result"]
        metrics_list = original_training_result["metrics_list"]
        original_best_epoch = str(original_training_result["best_epoch"])
        compressed_best_epoch = str(compressed_training_result["best_epoch"])
        original_best_metrics = list(original_training_result["valid_metrics"][original_best_epoch].values())
        compressed_best_metrics = list(compressed_training_result["valid_metrics"][compressed_best_epoch].values())

        fig, axs = plt.subplots(ncols=len(metrics_list), figsize=(15, 6))

        for idx, metric in enumerate(metrics_list):
            bars_original = Plotter._plot_single_bar(
                axs[idx], "Original Model", original_best_metrics[idx], "slategray"
            )
            bars_compressed = Plotter._plot_single_bar(
                axs[idx], "Compressed Model", compressed_best_metrics[idx], "dodgerblue"
            )

            for bar in bars_original:
                Plotter._add_value_annotations(axs[idx], bar, original_best_metrics[idx])

            for bar in bars_compressed:
                Plotter._add_value_annotations(axs[idx], bar, compressed_best_metrics[idx])

            Plotter._add_difference_annotations(
                axs[idx],
                original_best_metrics[idx],
                compressed_best_metrics[idx],
                compressed_best_metrics[idx] - original_best_metrics[idx],
            )

            Plotter._set_common_plot_settings(axs[idx], metric)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def compare_profile_result(profile_result):
        y_labels = ["FLOPs(M)", "Num of Params(M)", "Model Size(MB)"]
        keys = ["flops", "number_of_parameters", "size"]
        original_values = [profile_result["results"]["original_model"][_key] for _key in keys]
        compressed_values = [profile_result["results"]["compressed_model"][_key] for _key in keys]

        difference_values = np.array(original_values) / np.array(compressed_values)

        fig, axs = plt.subplots(ncols=len(y_labels), figsize=(15, 6))

        for idx, label in enumerate(y_labels):
            bars_original = axs[idx].bar(
                ["Original Model"],
                [original_values[idx]],
                color="slategray",
                label="Original Model",
                width=Plotter.BAR_WIDTH,
            )
            bars_compressed = axs[idx].bar(
                ["Compressed Model"],
                [compressed_values[idx]],
                color="dodgerblue",
                label="Compressed Model",
                width=Plotter.BAR_WIDTH,
            )

            for bar in bars_original:
                axs[idx].text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{bar.get_height():.4f}",
                    ha="center",
                    va="bottom",
                )

            for bar in bars_compressed:
                axs[idx].text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{bar.get_height():.4f}",
                    ha="center",
                    va="bottom",
                )

            axs[idx].scatter(
                ["Original Model", "Compressed Model"],
                [original_values[idx], compressed_values[idx]],
                color="red",
                marker="o",
                zorder=3,
            )

            axs[idx].plot(
                ["Original Model", "Compressed Model"],
                [original_values[idx], compressed_values[idx]],
                color="red",
                linestyle="--",
                linewidth=2,
                zorder=2,
            )

            diff_x = np.mean(axs[idx].get_xlim())
            axs[idx].text(
                diff_x,
                original_values[idx] * 0.5,
                f"{difference_values[idx]:.4f}x",
                ha="center",
                va="bottom",
                color="black",
            )

            axs[idx].set_ylabel(label)
            axs[idx].legend()
            axs[idx].grid(axis="y")

        plt.tight_layout()
        plt.show()
