import unittest
import subprocess
import time


# Module path to experiment or tuple(path, [extra_config_for_sacred])
# MiniSMD server_id=1 points to dataset with 1000 elements instead of the default 500
EXPERIMENTS = [
    'baselines.train_iqr_ad',
    'baselines.train_oos_ad',
    'baselines.train_pca_ad',
    'baselines.train_wmd_ad',
    'baselines.train_knn',
    'baselines.train_kmeans',
    'baselines.train_eif',
    'baselines.train_iforest',
    'generative.gan.train_beatgan',
    'generative.gan.train_lstm_vae_gan',
    'generative.gan.train_madgan',
    'generative.gan.train_tadgan',
    'generative.vae.train_donut',
    'generative.vae.train_gmm_vae',
    'generative.vae.train_lstm_vae_park',
    'generative.vae.train_lstm_vae_soelch',
    'generative.vae.train_omni_anomaly',
    'generative.vae.train_sis_vae',
    'other.train_lstm_ae_ocsvm',
    'other.train_mtad_gat',
    ('other.train_ncad', ['dataset.ds_args.server_id=1']),
    'other.train_thoc',
    'prediction.train_gdn',
    'prediction.train_lstm_prediction_filonov',
    'prediction.train_lstm_prediction_malhotra',
    'prediction.train_tcn_prediction_he',
    'prediction.train_tcn_prediction_munir',
    'reconstruction.train_anomtransf',
    'reconstruction.train_dense_ae',
    'reconstruction.train_genad',
    'reconstruction.train_lstm_ae',
    'reconstruction.train_lstm_max_ae',
    'reconstruction.train_mscred',
    'reconstruction.train_stgat_mad',
    'reconstruction.train_tcn_ae',
    'reconstruction.train_untrained_lstm_ae',
    'reconstruction.train_usad',
]

TIMES_TAKEN = {}

# Base TestCase class for each experiment.
# Captures runtime for each experiment
class TestClassBase(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        duration = time.time() - self.startTime
        TIMES_TAKEN[self.__class__.__name__] = duration


def make_test_function(experiment, extra_config):
    if not extra_config:
        extra_config = []
    def test(self):
        run = subprocess.run(['python', '-m', f'timesead_experiments.{experiment}',
                'with', 'dataset.name="MiniSMDDataset"', 'training.epochs=2'] + extra_config,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(run.stdout)
        self.assertEqual(run.returncode, 0, f'{experiment} failed to run!')
    return test


if __name__ == '__main__':

    # Generate different test cases for each experiment
    # Each experiment test will be named Test_<experiment_path> with _ instead of .
    # Example, Test_generative_gan_train_beatgan
    # Derived from: https://eli.thegreenplace.net/2014/04/02/dynamically-generating-python-test-cases
    for experiment in EXPERIMENTS:
        extra_config = None
        if isinstance(experiment, tuple):
            experiment, extra_config = experiment
        test_func = make_test_function(experiment, extra_config)
        experiment_name = experiment.replace('.', '_')
        classname = 'Test_{0}'.format(experiment_name)
        globals()[classname] = type(classname,
                                   (TestClassBase,),
                                   {f'test_gen_{experiment_name}': test_func})

    unittest.main(buffer=True, exit=False)

    print('Times taken:')
    for classname, duration in TIMES_TAKEN.items():
        print(f'{classname}: {duration:.3f}s')
    # print(f'Total time: {sum(TIMES_TAKEN.values()):.3f}s')
