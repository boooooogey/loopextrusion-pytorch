from encode_to_contact_utils import run_experiment

def patch_loop(main_folder, start_init, patch_num, resolution, patch_size, chromosome, mode):
    start = start_init
    for i in range(patch_num):
        print(f"Patch #{i}:")
        stop = start + resolution * patch_size#14400000

        experiment = os.path.join(main_folder, f'{chromosome}_{start}_{stop}')

        if not os.path.isdir(main_folder):
            os.mkdir(main_folder)
        if not os.path.isdir(experiment):
            os.mkdir(experiment)

        run_experiment(mode, chromosome, start, stop, resolution, experiment)

        start = stop

CHROMOSOME = "chr10"
START_INIT = 13600000
RESOLUTION = 8000
PATCH_SIZE = 150
PATCH_NUM = 20
MODE = "mean"
DIAG_STOP = 50
MAIN_FOLDER = ".local/experiment_results"

patch_loop(MAIN_FOLDER, START_INIT, PATCH_NUM, RESOLUTION, PATCH_SIZE, CHROMOSOME, MODE)