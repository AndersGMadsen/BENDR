import torch
import tqdm
import argparse
import os
import datetime

from dn3_ext import BendingCollegeWav2Vec, ConvEncoderBENDR, BENDRContextualizer
from dn3.configuratron import ExperimentConfig
from dn3.transforms.instance import To1020
from dn3.transforms.batch import RandomTemporalCrop

from torch.utils.data import ConcatDataset

# Since we are doing a lot of loading, this is nice to suppress some tedious information.
# Keep in mind, removing this might help debug data loading problems
import mne
mne.set_log_level(False)


def load_datasets(experiment):
    training = list()
    validation = None
    total_thinkers = 0
    for name, ds in experiment.datasets.items():
        print("Constructing " + name)
        dataset = ds.auto_construct_dataset()
        dataset.add_transform(To1020())
        if hasattr(experiment, 'validation_dataset') and experiment.validation_dataset == name:
            validation = dataset
            continue
        total_thinkers += len(dataset.get_thinkers())

        training.append(dataset)

    print("Training BENDR using {} people's data across {} datasets.".format(total_thinkers, len(training)))
    return ConcatDataset(training), validation, total_thinkers


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrains a BENDER model.")
    parser.add_argument('--config', default="configs/pretraining.yml", help="The DN3 config file to use.")
    parser.add_argument('--hidden-size', default=512, type=int, help="The hidden size of the encoder.")
    parser.add_argument('--resume', default=None, type=int, help="Whether to continue training the encoder from the "
                                                                 "specified epoch.")
    parser.add_argument('--num-workers', default=2, type=int)
    parser.add_argument('--no-save', action='store_true', help="Don't save checkpoints while training.")
    parser.add_argument('--no-save-epochs', action='store_true', help="Don't save epoch checkpoints while training")
    parser.add_argument('--results-folder', default=None, help='Name of the folder for final results.')
    return parser.parse_args()


if __name__ == '__main__':
    # Clear CUDA memory
    torch.cuda.empty_cache()
    
    # Get the current date and time like YYMMDD_HHMMSS string
    now = datetime.datetime.now()
    now_str = now.strftime("%y%m%d_%H%M%S")    
    
    args = parse_args()
    experiment = ExperimentConfig(args.config)

    training, validation, target_thinkers = load_datasets(experiment)

    encoder = ConvEncoderBENDR(len(To1020.EEG_20_div) + 1, encoder_h=args.hidden_size)
    tqdm.tqdm.write(encoder.description(experiment.global_sfreq, experiment.global_samples))
    contextualizer = BENDRContextualizer(encoder.encoder_h, layer_drop=experiment.bending_college_args.layer_drop)
    
    if args.resume is not None:
        encoder.load('checkpoints/encoder_epoch_{}_{}.pt'.format(args.resume, now_str))
        contextualizer.load('checkpoints/contextualizer_epoch_{}_{}.pt'.format(args.resume, now_str))

    process = BendingCollegeWav2Vec(encoder, contextualizer, **experiment.bending_college_args)

    # Slower learning rate for the encoder
    process.set_optimizer(torch.optim.Adam(process.parameters(), **experiment.optimizer_params))
    process.add_batch_transform(RandomTemporalCrop(max_crop_frac=experiment.augmentation_params.batch_crop_frac))

    tqdm.tqdm.write(process.description(experiment.global_samples))

    def epoch_checkpoint(metrics):
        if not args.no_save and not args.no_save_epochs:
            tqdm.tqdm.write("Saving...")
            encoder.save('checkpoints/encoder_epoch_{}_{}.pt'.format(metrics['epoch'], now_str))
            contextualizer.save('checkpoints/contextualizer_epoch_{}_{}.pt'.format(metrics['epoch'], now_str))
            
    def simple_checkpoint(metrics):
        if metrics is not None and metrics['Accuracy'] > experiment.mask_threshold and \
                metrics['Mask_pct'] < experiment.mask_pct_max:
            process.mask_span = int(process.mask_span * experiment.mask_inflation)
            tqdm.tqdm.write("Increased mask span to {} samples".format(process.mask_span))
        if not args.no_save:
            tqdm.tqdm.write("Saving...")
            encoder.save('checkpoints/encoder_{}.pt'.format(now_str))
            contextualizer.save('checkpoints/contextualizer_{}.pt'.format(now_str))

    # Slower learning rate for the encoder
    process.set_optimizer(torch.optim.Adam(process.parameters(), **experiment.optimizer_params))
    process.add_batch_transform(RandomTemporalCrop(max_crop_frac=experiment.augmentation_params.batch_crop_frac))

    tqdm.tqdm.write(process.description(experiment.global_samples))

    simple_checkpoint(None)

    train_log, validation_log = process.fit(training, epoch_callback=epoch_checkpoint, num_workers=args.num_workers,
                validation_dataset=validation, resume_epoch=args.resume, log_callback=simple_checkpoint,
                **experiment.training_params)

    print(process.evaluate(validation))

    if not args.no_save:
        tqdm.tqdm.write("Saving best model...")
        encoder.save('checkpoints/encoder_best_val_{}.pt'.format(now_str))
        contextualizer.save('checkpoints/contextualizer_best_val_{}.pt'.format(now_str))
    
    if args.results_folder is not None:
        train_log.to_csv(os.path.join(args.results_folder, 'pretrain_train_log_{}.csv'.format(now_str)))
        validation_log.to_csv(os.path.join(args.results_folder, 'pretrain_validation_log_{}.csv'.format(now_str)))
        
    print("Completed without error.")