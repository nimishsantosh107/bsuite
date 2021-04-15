from tabulate import tabulate

from bsuite.logging import csv_load
from bsuite.experiments import summary_analysis
from bsuite.experiments.catch import analysis as catch_analysis
from bsuite.experiments.catch_noise import analysis as catch_noise_analysis
from bsuite.experiments.cartpole import analysis as cartpole_analysis
from bsuite.experiments.cartpole_noise import analysis as cartpole_noise_analysis
from bsuite.experiments.mountain_car import analysis as mountaincar_analysis
from bsuite.experiments.mountain_car_noise import analysis as mountaincar_noise_analysis

class Analyzer:

    def __init__(self, results_dir):
        self.DF, self.SWEEP_VARS = csv_load.load_bsuite( {'Agent': results_dir} )
        self.BSUITE_SCORE = summary_analysis.bsuite_score(self.DF, self.SWEEP_VARS)
        self.BSUITE_SUMMARY = summary_analysis.ave_score_by_tag(self.BSUITE_SCORE, self.SWEEP_VARS)
        self.CATCH_DF = self.DF[self.DF.bsuite_env == 'catch'].copy()
        self.CATCH_NOISE_DF = self.DF[self.DF.bsuite_env == 'catch_noise'].copy()
        self.CARTPOLE_DF = self.DF[self.DF.bsuite_env == 'cartpole'].copy()
        self.CARTPOLE_NOISE_DF = self.DF[self.DF.bsuite_env == 'cartpole_noise'].copy()
        self.MOUNTAINCAR_DF = self.DF[self.DF.bsuite_env == 'mountain_car'].copy()
        self.MOUNTAINCAR_NOISE_DF = self.DF[self.DF.bsuite_env == 'mountain_car_noise'].copy()

    def plot(self):
        summary_analysis.bsuite_bar_plot(self.BSUITE_SCORE, self.SWEEP_VARS).draw();

    def get_scores(self):
        SCORE_CATCH = catch_analysis.score(self.CATCH_DF)
        SCORE_CATCH_NOISE = catch_noise_analysis.score(self.CATCH_NOISE_DF)
        SCORE_CARTPOLE = cartpole_analysis.score(self.CARTPOLE_DF)
        SCORE_CARTPOLE_NOISE = cartpole_noise_analysis.score(self.CARTPOLE_NOISE_DF)
        SCORE_MOUNTAINCAR = mountaincar_analysis.score(self.MOUNTAINCAR_DF)
        SCORE_MOUNTAINCAR_NOISE = mountaincar_noise_analysis.score(self.MOUNTAINCAR_NOISE_DF)

        return {
            'catch': SCORE_CATCH,
            'catch_noise': SCORE_CATCH_NOISE,
            'cartpole': SCORE_CARTPOLE,
            'cartpole_noise': SCORE_CARTPOLE_NOISE,
            'mountain_car': SCORE_MOUNTAINCAR,
            'mountain_car_noise': SCORE_MOUNTAINCAR_NOISE
        }
    
    def print_scores(self):
        scores = self.get_scores()

        headers = ["ENVIRONMENT", "SCORE"]
        table = tabulate([(env, score) for env, score in scores.items()], headers=headers, tablefmt="fancy_grid")
        print(table)