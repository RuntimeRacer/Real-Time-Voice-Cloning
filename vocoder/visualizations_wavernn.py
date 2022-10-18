from vocoder.vocoder_dataset import VocoderDataset
from datetime import datetime
from time import perf_counter as timer

import matplotlib
matplotlib.use("Agg")

import numpy as np
# import webbrowser
import visdom


class Visualizations:
    def __init__(self, env_name=None, update_every=10, server="http://localhost", disabled=False):
        # Tracking data
        self.last_update_timestamp = timer()
        self.update_every = update_every
        self.step_times = []
        self.losses = []
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
        self.loss_win = None
        # self.lr_win = None
        self.implementation_win = None
        self.implementation_string = ""

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

    def log_implementation(self, params):
        if self.disabled:
            return
        implementation_string = ""
        for param, value in params.items():
            implementation_string += "<b>%s</b>: %s\n" % (param, value)
            implementation_string = implementation_string.replace("\n", "<br>")
        self.implementation_string = implementation_string
        self.implementation_win = self.vis.text(
            implementation_string,
            opts={"title": "Training implementation"}
        )

    def update(self, loss, step):
        # Update the tracking data
        now = timer()
        self.step_times.append(1000 * (now - self.last_update_timestamp))
        self.last_update_timestamp = now
        self.losses.append(loss)
        print(".", end="")

        # Update the plots every <update_every> steps
        if step % self.update_every != 0:
            return
        time_string = "Step time:  mean: %5dms  std: %5dms" % \
                      (int(np.mean(self.step_times)), int(np.std(self.step_times)))
        print("\nStep %6d Loss: %.6f  %s" %
              (step, np.mean(self.losses), time_string))

        if not self.disabled:
            self.loss_win = self.vis.line(
                [np.mean(self.losses)],
                [step],
                win=self.loss_win,
                update="append" if self.loss_win else None,
                opts=dict(
                    legend=["Avg. loss"],
                    xlabel="Step",
                    ylabel="Loss",
                    title="Loss",
                )
            )

            if self.implementation_win is not None:
                self.vis.text(
                    self.implementation_string + ("<b>%s</b>" % time_string),
                    win=self.implementation_win,
                    opts={"title": "Training implementation"},
                )

        # Reset the tracking
        self.losses.clear()
        self.step_times.clear()

    def save(self):
        if not self.disabled:
            self.vis.save([self.env_name])
