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

        for idx, _metric in enumerate(metrics_list):
            labels = {
                "map50": "mAP@[.50]",
                "map75": "mAP@[.75]",
                "map50_95": "mAP@[.50:.95]",
            }
            metric = labels[_metric]
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

    @staticmethod
    def _plot_epoch_metrics(train_data, valid_data, title, xlabel, ylabel):
        labels = {
            "map50": "mAP@[.50]",
            "map75": "mAP@[.75]",
            "map50_95": "mAP@[.50:.95]",
        }
        metric = labels[ylabel]

        train_epochs = list(train_data.keys())
        valid_epochs = list(valid_data.keys())

        train_metric_values = [metric[ylabel] for _, metric in train_data.items()]
        valid_metric_values = [metric[ylabel] for _, metric in valid_data.items()]

        plt.figure(figsize=(15, 6))
        plt.plot(train_epochs, train_metric_values, label="Train Metric", marker="o", color="slategray")
        plt.plot(valid_epochs, valid_metric_values, label="Validation Metric", marker="x", color="dodgerblue")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True)
        max_value = max(train_metric_values + valid_metric_values)
        plt.ylim(0, max_value * 1.2)

        for epoch, metric in valid_data.items():
            metric = metric[ylabel]
            Plotter._add_text_annotation(plt.gca(), epoch, metric, metric)

        plt.show()

    @staticmethod
    def plot_metric_by_epoch(train_data, valid_data, title="Train and Validation metric per epoch", xlabel="Epochs"):
        for _ylabel in ["map50", "map75", "map50_95"]:
            Plotter._plot_epoch_metrics(train_data, valid_data, title, xlabel, _ylabel)

    @staticmethod
    def _plot_epoch_losses(train_data, valid_data, title, xlabel, ylabel):
        train_epochs = list(train_data.keys())
        valid_epochs = list(valid_data.keys())

        train_loss_values = list(train_data.values())
        valid_loss_values = list(valid_data.values())

        plt.figure(figsize=(15, 6))
        plt.plot(train_epochs, train_loss_values, label="Train Loss", marker="o", color="slategray")
        plt.plot(valid_epochs, valid_loss_values, label="Validation Loss", marker="x", color="dodgerblue")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        max_value = max(train_loss_values + valid_loss_values)
        plt.ylim(0, max_value * 1.2)

        for epoch, loss in valid_data.items():
            Plotter._add_text_annotation(plt.gca(), epoch, loss, loss)

        plt.show()

    @staticmethod
    def plot_loss_by_epoch(
        train_data, valid_data, title="Train and Validation loss per epoch", xlabel="Epochs", ylabel="Loss"
    ):
        Plotter._plot_epoch_losses(train_data, valid_data, title, xlabel, ylabel)

    @staticmethod
    def compare_by_step_size(
        a_compression_result, b_compression_result, a_benchmark_result, b_benchmark_result, x_labels=None
    ):
        if x_labels is None:
            x_labels = [2, 32]
        x_labels = [f"step_size={x_label}" for x_label in x_labels]
        y_labels = ["Latency(ms)", "FLOPs(M)", "Num of Params(M)", "Model Size(MB)"]
        keys = ["latency", "flops", "number_of_parameters", "size"]

        a_values = [a_benchmark_result["result"]["latency"]]
        b_values = [b_benchmark_result["result"]["latency"]]

        for _key in keys[1:]:
            a_values.append(a_compression_result["results"]["compressed_model"][_key])

        for _key in keys[1:]:
            b_values.append(b_compression_result["results"]["compressed_model"][_key])

        fig, axs = plt.subplots(ncols=len(y_labels), figsize=(15, 5))

        for idx, label in enumerate(y_labels):
            bars_original = axs[idx].bar(
                [x_labels[0]],
                [a_values[idx]],
                color="slategray",
                label=x_labels[0],
                width=Plotter.BAR_WIDTH,
            )
            bars_compressed = axs[idx].bar(
                [x_labels[1]],
                [b_values[idx]],
                color="dodgerblue",
                label=x_labels[1],
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

            axs[idx].set_ylabel(label)
            axs[idx].set_title(f"Step Size vs. {label}")
            axs[idx].legend()
            axs[idx].grid(axis="y")
            max_value = max(a_values[idx], b_values[idx])
            axs[idx].set_ylim(max_value * 0.8, max_value * 1.1)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_profile_results(data, x_labels, title_prefix, y_labels=None):
        if y_labels is None:
            y_labels = ["latency", "flops", "params", "size"]
        y_label_names = {
            "latency": "Latency(ms)",
            "flops": "FLOPs(M)",
            "params": "Num of Params(M)",
            "size": "Model Size(MB)",
        }

        fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(15, 5))

        for idx, label in enumerate(y_labels):
            y_label_name = y_label_names[label]
            axs[idx].plot(x_labels, data[label], marker="o", color="dodgerblue", label=label)

            axs[idx].set_xlabel("Step Size" if "Step" in title_prefix else "Compression Ratio")
            axs[idx].set_ylabel(y_label_name)
            axs[idx].set_title(f"{title_prefix} vs. {y_label_name}")
            axs[idx].grid(True)

            max_value = max(data[label])
            axs[idx].set_ylim(0, max_value * 1.2)
            axs[idx].set_xlim(-0.5, len(x_labels) - 0.5)

            for i, value in enumerate(data[label]):
                axs[idx].text(i, value, f"{value:.2f}", ha="center", va="bottom")

        plt.tight_layout()
        plt.show()

    @staticmethod
    def prepare_data(original_benchmark_result, compressed_benchmark_result, profile_result):
        # Original and compressed latency values
        original_latency = original_benchmark_result["result"]["latency"]
        compressed_latency = compressed_benchmark_result["result"]["latency"]
        original_values = [original_latency]
        compressed_values = [compressed_latency]

        # FLOPs, number of parameters, and model size
        keys = ["flops", "number_of_parameters", "size"]
        original_values.extend(profile_result["results"]["original_model"][_key] for _key in keys)
        compressed_values.extend(profile_result["results"]["compressed_model"][_key] for _key in keys)

        return original_values, compressed_values

    @staticmethod
    def summary_profile_results(original_benchmark_result, compressed_benchmark_result, profile_result):
        original_values, compressed_values = Plotter.prepare_data(
            original_benchmark_result, compressed_benchmark_result, profile_result
        )
        y_labels = ["Latency(ms)", "FLOPs(M)", "Num of Params(M)", "Model Size(MB)"]

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
            max_value = max([original_values[idx], compressed_values[idx]])
            axs[idx].set_ylim(0, max_value * 1.2)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def line_plot_overall_latencies(step_sizes, compression_ratios, latencies, original_latency, target_latency):
        plt.figure(figsize=(10, 6))

        for i, step_size in enumerate(step_sizes):
            plt.plot(compression_ratios, latencies[i], marker="o", label=f"Step Size={step_size}")
            for j, latency in enumerate(latencies[i]):
                plt.text(compression_ratios[j], latency, f"{latency:.2f}", ha="center", va="bottom")

        plt.xlabel("Compression Ratio")
        plt.ylabel("Latency (ms)")
        plt.title("Latency vs. Compression Ratio for Different Step Sizes")
        plt.axhline(
            original_latency, color="slategray", linestyle="--", label=f"Original Latency: {original_latency:.2f} ms"
        )
        if target_latency:
            plt.axhline(target_latency, color="red", linestyle="--", label=f"Target Latency: {target_latency:.2f} ms")
        plt.legend()
        plt.grid(True)
        plt.xticks(compression_ratios)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def bar_plot_overall_latencies(step_sizes, compression_ratios, latencies):
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(compression_ratios))
        width = 0.35

        for i, step_size in enumerate(step_sizes):
            bars = ax.bar(x + i * width, latencies[i], width, label=f"Step Size={step_size}")
            for bar in bars:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{bar.get_height():.2f}",
                    ha="center",
                    va="bottom",
                )

        ax.set_xlabel("Compression Ratio")
        ax.set_ylabel("Latency")
        ax.set_title("Latency for Different Compression Ratios and Step Sizes")
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(compression_ratios)
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def bar_plot_overall_latencies_by_device(model_names, latency_of_models, devices):
        for model, data in zip(model_names, latency_of_models):
            fig, ax = plt.subplots(figsize=(12, 6))
            bars = ax.bar(devices, data, color="dodgerblue")
            ax.set_xlabel("Devices")
            ax.set_ylabel("Latency")
            ax.set_title(f"Latency of {model} on Different Devices")
            ax.grid(True)
            plt.xticks(rotation=90)
            max_value = max(data)
            ax.set_ylim(0, max_value * 1.2)

            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.2f}", ha="center", va="bottom")

            plt.tight_layout()
            plt.show()
