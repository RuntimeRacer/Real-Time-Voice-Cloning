from encoder.data_objects.speaker_verification_dataset import SpeakerVerificationDataset
from datetime import datetime
from time import perf_counter as timer

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
# import webbrowser
import visdom
import umap

colormap = np.array([
    [33, 0, 127],
    [32, 25, 35],
    [0, 0, 255],
    [252, 255, 93],
    [125, 252, 0],
    [14, 196, 52],
    [34, 140, 104],
    [138, 216, 232],
    [35, 91, 84],
    [41, 189, 171],
    [57, 152, 245],
    [55, 41, 79],
    [39, 125, 167],
    [55, 80, 219],
    [242, 32, 32],
    [153, 25, 25],
    [255, 203, 165],
    [230, 143, 102],
    [197, 97, 51],
    [150, 52, 28],
    [99, 40, 25],
    [255, 196, 19],
    [244, 122, 34],
    [47, 42, 160],
    [183, 50, 204],
    [119, 43, 157],
    [240, 124, 171],
    [211, 11, 148],
    [237, 239, 243],
    [195, 165, 180],
    [148, 106, 162],
    [93, 76, 134],
    [0, 0, 0],
    [183, 183, 183],
    [222,184,135],
    [178,34,34],
    [0,0,139],
    [210,105,30],
    [65,105,225],
    [255,255,240],
    [255,239,213],
    [165,42,42],
    [128,128,128],
    [143,188,143],
    [127,255,0],
    [255,127,80],
    [102,205,170],
    [186,85,211],
    [0,128,128],
    [176,196,222],
    [128,128,0],
    [255,182,193],
    [135,206,250],
    [139,69,19],
    [60,179,113],
    [0,0,205],
    [255,20,147],
    [0,255,255],
    [255,235,205],
    [248,248,255],
    [124,252,0],
    [210,180,140],
    [100,149,237],
    [255,228,181],
    [0,0,128],
    [64,224,208],
    [144,238,144],
    [0,255,255],
    [255,105,180],
    [245,222,179],
    [139,0,139],
    [255,255,224],
    [128,0,128],
    [245,245,220],
    [30,144,255],
    [218,165,32],
    [0,250,154],
    [211,211,211],
    [34,139,34],
    [0,139,139],
    [0,128,0],
    [47,79,79],
    [0,255,127],
    [189,183,107],
    [250,128,114],
    [176,224,230],
    [233,150,122],
    [255,228,196],
    [250,240,230],
    [255,228,225],
    [255,160,122],
    [112,128,144],
    [255,0,255],
    [46,139,87],
    [148,0,211],
    [173,255,47],
    [238,130,238],
    [138,43,226],
    [70,130,180],
    [255,240,245],
    [205,133,63],
    [0,255,0],
    [107,142,35],
    [135,206,235],
    [255,192,203],
    [154,205,50],
    [255,255,0],
    [244,164,96],
    [224,255,255],
    [139,0,0],
    [240,230,140],
    [188,143,143],
    [240,255,240],
    [230,230,250],
    [250,235,215],
    [32,178,170],
    [255,69,0],
    [0,0,255],
    [255,245,238],
    [152,251,152],
    [220,220,220],
    [192,192,192],
    [0,100,0],
    [169,169,169],
    [238,232,170],
    [219,112,147],
    [255,215,0],
    [95,158,160],
    [240,128,128],
    [119,136,153],
    [184,134,11],
    [216,191,216],
    [250,250,210],
    [123,104,238],
    [255,250,250],
    [160,82,45],
    [255,165,0],
    [245,255,250],
    [153,50,204],
    [25,25,112],
    [199,21,133],
    [72,209,204],
    [72,61,139],
    [128,0,0],
    [240,255,255],
    [255,250,205],
    [253,245,230],
    [175,238,238],
    [173,216,230],
    [221,160,221],
    [255,99,71],
    [255,140,0],
    [106,90,205],
    [205,92,92],
    [255,218,185],
    [105,105,105],
    [240,248,255],
    [220,20,60],
    [255,222,173],
    [255,248,220],
    [255,0,0],
    [0,0,0],
    [218,112,214],
    [0,191,255],
    [127,255,212],
    [255,250,240],
    [85,107,47],
    [50,205,50],
    [0,206,209],
    [147,112,219],
    [75,0,130],
    [33, 0, 127],
    [32, 25, 35],
    [0, 0, 255],
    [252, 255, 93],
    [125, 252, 0],
    [14, 196, 52],
    [34, 140, 104],
    [138, 216, 232],
    [35, 91, 84],
    [41, 189, 171],
    [57, 152, 245],
    [55, 41, 79],
    [39, 125, 167],
    [55, 80, 219],
    [242, 32, 32],
    [153, 25, 25],
    [255, 203, 165],
    [230, 143, 102],
    [197, 97, 51],
    [150, 52, 28],
    [99, 40, 25],
    [255, 196, 19],
    [244, 122, 34],
    [47, 42, 160],
    [183, 50, 204],
    [119, 43, 157],
    [240, 124, 171],
    [211, 11, 148],
    [237, 239, 243],
    [195, 165, 180],
    [148, 106, 162],
    [93, 76, 134],
    [0, 0, 0],
    [183, 183, 183],
    [222,184,135],
    [178,34,34],
    [0,0,139],
    [210,105,30],
    [65,105,225],
    [255,255,240],
    [255,239,213],
    [165,42,42],
    [128,128,128],
    [143,188,143],
    [127,255,0],
    [255,127,80],
    [102,205,170],
    [186,85,211],
    [0,128,128],
    [176,196,222],
    [128,128,0],
    [255,182,193],
    [135,206,250],
    [139,69,19],
    [60,179,113],
    [0,0,205],
    [255,20,147],
    [0,255,255],
    [255,235,205],
    [248,248,255],
    [124,252,0],
    [210,180,140],
    [100,149,237],
    [255,228,181],
    [0,0,128],
    [64,224,208],
    [144,238,144],
    [0,255,255],
    [255,105,180],
    [245,222,179],
    [139,0,139],
    [255,255,224],
    [128,0,128],
    [245,245,220],
    [30,144,255],
    [218,165,32],
    [0,250,154],
    [211,211,211],
    [34,139,34],
    [0,139,139],
    [0,128,0],
    [47,79,79],
    [0,255,127],
    [189,183,107],
    [250,128,114],
    [176,224,230],
    [233,150,122],
    [255,228,196],
    [250,240,230],
    [255,228,225],
    [255,160,122],
    [112,128,144],
    [255,0,255],
    [46,139,87],
    [148,0,211],
    [173,255,47],
    [238,130,238],
    [138,43,226],
    [70,130,180],
    [255,240,245],
    [205,133,63],
    [0,255,0],
    [107,142,35],
    [135,206,235],
    [255,192,203],
    [154,205,50],
    [255,255,0],
    [244,164,96],
    [224,255,255],
    [139,0,0],
    [240,230,140],
    [188,143,143],
    [240,255,240],
    [230,230,250],
    [250,235,215],
    [32,178,170],
    [255,69,0],
    [0,0,255],
    [255,245,238],
    [152,251,152],
    [220,220,220],
    [192,192,192],
    [0,100,0],
    [169,169,169],
    [238,232,170],
    [219,112,147],
    [255,215,0],
    [95,158,160],
    [240,128,128],
    [119,136,153],
    [184,134,11],
    [216,191,216],
    [250,250,210],
    [123,104,238],
    [255,250,250],
    [160,82,45],
    [255,165,0],
    [245,255,250],
    [153,50,204],
    [25,25,112],
    [199,21,133],
    [72,209,204],
    [72,61,139],
    [128,0,0],
    [240,255,255],
    [255,250,205],
    [253,245,230],
    [175,238,238],
    [173,216,230],
    [221,160,221],
    [255,99,71],
    [255,140,0],
    [106,90,205],
    [205,92,92],
    [255,218,185],
    [105,105,105],
    [240,248,255],
    [220,20,60],
    [255,222,173],
    [255,248,220],
    [255,0,0],
    [0,0,0],
    [218,112,214],
    [0,191,255],
    [127,255,212],
    [255,250,240],
    [85,107,47],
    [50,205,50],
    [0,206,209],
    [147,112,219],
    [75,0,130],
], dtype=np.float) / 255


class Visualizations:
    def __init__(self, env_name=None, update_every=10, server="http://localhost", disabled=False):
        # Tracking data
        self.last_update_timestamp = timer()
        self.update_every = update_every
        self.step_times = []
        self.losses = []
        self.eers = []
        self.lang_losses = []
        self.lang_eers = []
        self.age_losses = []
        self.age_eers = []
        self.sex_losses = []
        self.sex_eers = []
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
            raise Exception("No visdom server detected. Run the command \"visdom\" in your CLI to "
                            "start it.")
        # webbrowser.open("http://localhost:8097/env/" + self.env_name)

        # Create the windows
        self.loss_win = None
        self.eer_win = None
        self.lang_loss_win = None
        self.lang_eer_win = None
        self.age_loss_win = None
        self.age_eer_win = None
        self.sex_loss_win = None
        self.sex_eer_win = None
        # self.lr_win = None
        self.implementation_win = None
        self.projection_win = None
        self.lang_projection_win = None
        self.sex_projection_win = None
        self.age_projection_win = None
        self.implementation_string = ""

    def log_params(self):
        if self.disabled:
            return
        from encoder import params_data
        from encoder import params_model
        param_string = "<b>Model parameters</b>:<br>"
        for param_name in (p for p in dir(params_model) if not p.startswith("__")):
            value = getattr(params_model, param_name)
            param_string += "\t%s: %s<br>" % (param_name, value)
        param_string += "<b>Data parameters</b>:<br>"
        for param_name in (p for p in dir(params_data) if not p.startswith("__")):
            value = getattr(params_data, param_name)
            param_string += "\t%s: %s<br>" % (param_name, value)
        self.vis.text(param_string, opts={"title": "Parameters"})

    def log_dataset(self, dataset: SpeakerVerificationDataset):
        if self.disabled:
            return
        dataset_string = ""
        dataset_string += "<b>Speakers</b>: %s\n" % len(dataset.speakers)
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

    def update(self, loss, acc, step):
        # Update the tracking data
        now = timer()
        self.step_times.append(1000 * (now - self.last_update_timestamp))
        self.last_update_timestamp = now
        self.losses.append(loss)
        self.eers.append(acc)
        print(".", end="")

        # Update the plots every <update_every> steps
        if step % self.update_every != 0:
            return
        time_string = "Step time:  mean: %5dms  std: %5dms" % \
                      (int(np.mean(self.step_times)), int(np.std(self.step_times)))
        print("\nStep %6d Loss: %.4f ACC: %.4f  %s" %
              (step, np.mean(self.losses), np.mean(self.eers), time_string))

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
            self.eer_win = self.vis.line(
                [np.mean(self.eers)],
                [step],
                win=self.eer_win,
                update="append" if self.eer_win else None,
                opts=dict(
                    legend=["Avg. ACC"],
                    xlabel="Step",
                    ylabel="ACC",
                    title="Accuracy"
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
        self.eers.clear()
        self.step_times.clear()

    def draw_projections(self, embeds, utterances_per_speaker, step, out_fpath=None, max_speakers=342):
        max_speakers = min(max_speakers, len(colormap))
        embeds = embeds[:max_speakers * utterances_per_speaker]

        n_speakers = len(embeds) // utterances_per_speaker
        ground_truth = np.repeat(np.arange(n_speakers), utterances_per_speaker)
        colors = [colormap[i] for i in ground_truth]

        #reducer = umap.UMAP(min_dist=0.0, metric='cosine')
        reducer = umap.UMAP(min_dist=0.0)
        projected = reducer.fit_transform(embeds)
        plt.figure(figsize=(10,10), dpi=80)
        plt.scatter(projected[:, 0], projected[:, 1], c=colors, alpha=0.6)

        # plot limits
        # plt.gca().set_aspect("equal", "datalim")
        plt.xlim(-20, 20)
        plt.ylim(-20, 20)

        # plot title
        plt.title("UMAP projection (step {}, speakers: {})".format(step, n_speakers))

        # does this work?
        plt.tight_layout()

        if not self.disabled:
            self.projection_win = self.vis.matplot(plt, win=self.projection_win)
        if out_fpath is not None:
            plt.savefig(out_fpath, bbox_inches='tight', pad_inches=0.01)
        plt.close()
        plt.clf()

    def save(self):
        if not self.disabled:
            self.vis.save([self.env_name])
