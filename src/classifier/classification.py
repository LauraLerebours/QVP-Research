import os
import sys
sys.path.append('C:Users\\lereb\\QVP-Research\\src\\classifier')
from inference import classifier_inference_example
from preprocessing import find_sorted_classifier_samples, \
    find_classifier_samples, calc_mel_spec_example
from training import classifier_training_example

if __name__ == '__main__':
    samples_root_dir = 'C:\\Users\\lereb\\QVP-Research\\src\\data\\input\\Train\\B'
    samples_save_path = '...\\data\\samples.npz'
    samples_are_sorted = True
    print('sorting samples')
    if samples_are_sorted:
        # Use this if samples are sorted into folders like in the example data;
        # folder structure must follow the same naming convention as the example
        find_sorted_classifier_samples(samples_root_dir, samples_save_path)
    else:
        # Use this if you want to crawl a large, unorganized sample library.
        # Assigned labels will be less reliable
        find_classifier_samples(samples_root_dir, samples_save_path)
    print('calculating mel spectrograms')
    # Calculate Mel spectrograms and classes for the found samples
    calc_mel_spec_example(samples_save_path)

    mels_path = '..\\..\\data\\input\\classifier\\' \
                'mels_sr16000_hl256_nm128_nf1024_mls32767_naT_nmT.npz'

    # Train classifier model
    classifier_training_example(mels_path)

    # Change model name to match the best one that was trained
    model_name = 'class_cnn3_e17_vl0.72_vlacc0.7590.h5'

    test_sample_names = ['B21.wav',
                         'C27.wav',
                         'T41.wav']

    # Demonstrate classifier model on un-seen samples
    for test_sample_name in test_sample_names:
        test_sample_path = os.path.join(
            '..\\..\\data\\input\\Test', test_sample_name)

        classifier_inference_example(model_name, test_sample_path)