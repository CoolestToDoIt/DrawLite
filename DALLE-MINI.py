import wandb
run = wandb.init()
artifact = run.use_artifact('dalle-mini/dalle-mini/mega-1:latest')
artifact_dir = artifact.download()