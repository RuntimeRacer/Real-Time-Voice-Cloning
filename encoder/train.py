from encoder.visualizations import Visualizations
from encoder.data_objects import SpeakerVerificationDataLoader, SpeakerVerificationDataset
from encoder.params_model import *
from encoder.model import SpeakerEncoder
from utils.profiler import Profiler
from pathlib import Path
from accelerate import Accelerator
import torch
    

def train(run_id: str, clean_data_root: Path, models_dir: Path, umap_every: int, save_every: int,
          backup_every: int, vis_every: int, profile_every: int, force_restart: bool, visdom_server: str,
          no_visdom: bool, threads: int, end_after: int):

    # Initialize Accelerator
    accelerator = Accelerator()

    if accelerator.is_local_main_process:
        print("Accelerator process count: {0}".format(accelerator.num_processes))

    # Create a dataset and a dataloader
    dataset = SpeakerVerificationDataset(clean_data_root)
    loader = SpeakerVerificationDataLoader(
        dataset,
        speakers_per_batch,
        utterances_per_speaker,
        num_workers=threads,
        pin_memory=True
    )

    # Setup the device on which to run the forward pass and the loss. These can be different,
    # because the forward pass is faster on the GPU whereas the loss is often (depending on your
    # hyperparameters) faster on the CPU.
    device = accelerator.device

    # Create the model and the optimizer
    model = SpeakerEncoder(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_init)

    # Configure file path for the model
    state_fpath = models_dir.joinpath(run_id + ".pt")
    backup_dir = models_dir.joinpath(run_id + "_backups")

    # Initialize the model if not initialized yet
    if force_restart or not state_fpath.exists():
        accelerator.wait_for_everyone()
        with accelerator.local_main_process_first():
            if accelerator.is_local_main_process:
                print("Starting the training from scratch.")
                save(accelerator, model, state_fpath, 0)

    # Model has been initialized - Load the weights
    print("{0} - Loading weights at {1}".format(device, state_fpath))
    load(model, device, state_fpath, optimizer)
    current_step = model.step + 1
    print("{0} - Encoder weights loaded from step {1}".format(device, current_step))

    # Apply learning rate
    # optimizer.param_groups[0]["lr"] = learning_rate_init
    for p in optimizer.param_groups:
        p["lr"] = learning_rate_init

    # Set model in training mode
    model.train()

    # Initialize the visualization environment
    vis = Visualizations(run_id, vis_every, server=visdom_server, disabled=no_visdom)
    if accelerator.is_local_main_process:
        vis.log_dataset(dataset)
        vis.log_params()
        # FIXME: Print all device names in case we got multiple GPUs or CPUs
        if accelerator.state.num_processes > 1:
            vis.log_implementation({"Devices": str(accelerator.state.num_processes)})
        else:
            device_name = str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
            vis.log_implementation({"Device": device_name})

    # Accelerator code - optimize and prepare model
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    # Profiling only in main thread for comprehensibility
    if accelerator.is_local_main_process:
        profiler = Profiler(summarize_every=profile_every, disabled=False)

    # Training loop
    for step, speaker_batch in enumerate(loader, current_step):
        if accelerator.is_local_main_process:
            profiler.tick("Blocking, waiting for batch (threaded)")

        # Forward pass
        inputs = torch.from_numpy(speaker_batch.data)
        if accelerator.is_local_main_process:
            profiler.tick("Data to %s" % device)

        embeds = model(inputs)
        if accelerator.is_local_main_process:
            profiler.tick("Forward pass")

        embeds_loss = embeds.view((speakers_per_batch, utterances_per_speaker, -1))
        loss, eer = model.module.loss(embeds_loss)
        if accelerator.is_local_main_process:
            profiler.tick("Loss")

        # Backward pass
        model.module.zero_grad()
        accelerator.backward(loss)
        if accelerator.is_local_main_process:
            profiler.tick("Backward pass")

        model.module.do_gradient_ops(accelerator)
        optimizer.step()
        if accelerator.is_local_main_process:
            profiler.tick("Parameter update")

        # Update visualizations
        # learning_rate = optimizer.param_groups[0]["lr"]
        vis.update(loss.item(), eer, step)

        # Save visdom values
        if accelerator.is_local_main_process and vis_every != 0 and step % vis_every == 0 and (umap_every == 0 or step % umap_every != 0):
            vis.save()

        # Draw projections and save them to the backup folder
        if accelerator.is_local_main_process and umap_every != 0 and step % umap_every == 0:
            print("Drawing and saving projections (step %d)" % step)
            backup_dir.mkdir(exist_ok=True)
            projection_fpath = backup_dir.joinpath("%s_umap_%06d.png" % (run_id, step))
            embeds = embeds.detach().cpu().numpy()
            vis.draw_projections(embeds, utterances_per_speaker, step, projection_fpath)
            vis.save()

        # Overwrite the latest version of the model
        if save_every != 0 and step % save_every == 0:
            # Accelerator: Save in main process after sync
            accelerator.wait_for_everyone()
            with accelerator.local_main_process_first():
                if accelerator.is_local_main_process:
                    print("Saving the model (step %d)" % step)
                    save(accelerator, model, state_fpath, step, optimizer)

        # Make a backup
        if backup_every != 0 and step % backup_every == 0:
            # Accelerator: Save in main process after sync
            accelerator.wait_for_everyone()
            with accelerator.local_main_process_first():
                if accelerator.is_local_main_process:
                    print("Making a backup (step %d)" % step)
                    backup_dir.mkdir(exist_ok=True)
                    backup_fpath = backup_dir.joinpath("%s_bak_%06d.pt" % (run_id, step))
                    save(accelerator, model, backup_fpath, step, optimizer)

        if accelerator.is_local_main_process:
            profiler.tick("Extras (visualizations, saving)")

        # End training
        if end_after != 0 and step % end_after == 0:
            # Accelerator: Save in main process after sync
            accelerator.wait_for_everyone()
            with accelerator.local_main_process_first():
                if accelerator.is_local_main_process:
                    print("Step %d processed. Ending training." % step)
                    save(accelerator, model, state_fpath, step, optimizer)
            break

def save(accelerator, model, path, step, optimizer=None):
    # Unwrap Model
    model = accelerator.unwrap_model(model)

    # Save
    if optimizer is not None:
        torch.save({
            "step": step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }, str(path))
    else:
        torch.save({
            "step": step,
            "model_state": model.state_dict(),
        }, str(path))

def load(model, device, path, optimizer=None):
    # Use device of model params as location for loaded state
    checkpoint = torch.load(str(path), map_location=device)

    # Load model state
    model.load_state_dict(checkpoint["model_state"])
    model.step = checkpoint["step"]

    # Load optimizer state
    if "optimizer_state" in checkpoint and optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])