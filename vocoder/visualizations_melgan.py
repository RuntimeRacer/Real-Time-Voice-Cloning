from vocoder.vocoder_dataset import VocoderDataset
from datetime import datetime
from time import perf_counter as timer
import numpy as np
import visdom
import matplotlib
matplotlib.use("Agg")


class Visualizations:
    def __init__(self, env_name=None, update_every=10, server="http://localhost", disabled=False):
        # Tracking data
        self.last_update_timestamp = timer()
        self.update_every = update_every
        self.step_times = []
        self.losses = {}
        print("Updating the visualizations every %d steps." % update_every)

        # If visdom is disabled TODO: use a better paradigm for that
        self.disabled = disabled
        if self.disabled:
            return

        # Set the environment name
        now = str(datetime.now().strftime("%d-%m %Hh%M"))
        if env_name is None:
            self.env_name = now
        else:
            self.env_name = "%s (%s)" % (env_name, now)

        # Connect to visdom and open the corresponding window in the browser
        try:
            self.vis = visdom.Visdom(server, env=self.env_name, raise_exceptions=True)
        except ConnectionError:
            raise Exception("No visdom server detected. Run the command \"visdom\" in your CLI to start it.")
        # webbrowser.open("http://localhost:8097/env/" + self.env_name)

        # Create the windows
        self.loss_wins = {}
        # self.lr_win = None

    def log_params(self, vocoder_hparams):
        if self.disabled:
            return
        param_string = "<b>Training parameters</b>:<br>"
        for param_name in (p for p in dir(vocoder_hparams) if not p.startswith("__")):
            value = getattr(vocoder_hparams, param_name)
            param_string += "\t%s: %s<br>" % (param_name, value)
        self.vis.text(param_string, opts={"title": "Parameters"})

    def log_dataset(self, dataset: VocoderDataset):
        if self.disabled:
            return
        dataset_string = ""
        dataset_string += "<b>Samples</b>: %s\n" % len(dataset.samples_fpaths)
        dataset_string += "\n" + dataset.get_logs()
        dataset_string = dataset_string.replace("\n", "<br>")
        self.vis.text(dataset_string, opts={"title": "Dataset"})

    def update(self, loss_dict, step):
        # Update the tracking data
        now = timer()
        self.step_times.append(1000 * (now - self.last_update_timestamp))
        self.last_update_timestamp = now
        self._append_losses(loss_dict)
        print(".", end="")

        # Update the plots every <update_every> steps
        if step % self.update_every != 0:
            return

        # Print update for all losses
        self._print_mean_losses(step)

        if not self.disabled:
            self._update_windows(step)

        # Reset the tracking
        self.losses.clear()
        self.step_times.clear()

    def save(self):
        if not self.disabled:
            self.vis.save([self.env_name])

    def _append_losses(self, loss_dict):
        for k, v in loss_dict.items():
            if k not in self.losses:
                self.losses[k] = []
            self.losses[k].append(v)

    def _print_mean_losses(self, step):
        time_string = "Step time:  mean: %5dms  std: %5dms" % (
            int(np.mean(self.step_times)), int(np.std(self.step_times))
        )
        for k, v in self.losses.items():
            print("\nStep %6d %s: %.6f  %s" % (step, k, np.mean(v), time_string))

    def _update_windows(self, step):
        for k, v in self.losses.items():
            self.loss_wins[k] = self.vis.line(
                [np.mean(v)],
                [step],
                win=self.loss_wins[k] if k in self.loss_wins else None,
                update="append" if k in self.loss_wins else None,
                opts=dict(
                    legend=["Avg. %s" % k],
                    xlabel="Step",
                    ylabel="Loss",
                    title=k,
                )
            )

