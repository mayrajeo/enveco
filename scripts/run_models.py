from fastai.data.all import *
from fastai.tabular.all import *
absolute_path = Path().absolute()
sys.path.append(str(absolute_path.parents[0]))

from enveco.tabular.preprocessing import *
from enveco.models.ensemble import *
from enveco.interpretation import *
from enveco.metrics import *

from sklearn.ensemble import RandomForestRegressor

def run_models():
    target_variables = ['v', 'g', 'h', 'd']

    data_path = Path('../../enveco_data/enveco/')

    preprocessor = EnvecoPreprocessor(data_path/'AV.leaf.on.train.csv', 
                                      data_path/'AV.leaf.on.val.csv',
                                      data_path/'AV.leaf.on.test.csv')
    for t in target_variables:
        if not os.path.exists(t): os.makedirs(t)
        trainval_tb, test_tb = preprocessor.preprocess_lidar(target_col='v', path=data_path/'AV_las/', 
                                                             min_h=1.5,
                                                             height_features=True,
                                                             point_features=True, 
                                                             intensity_features=True, 
                                                             height_quantiles=True,
                                                             point_proportions=True, 
                                                             canopy_densities=True,
                                                             normalize=True,
                                                             log_y=False)

        dls = trainval_tb.dataloaders(bs=64, y_block=RegressionBlock())

        y_max = trainval_tb.train.y.max() * 1.1

        ensemble = Ensemble(dls, learn_func=tabular_learner,
                            y_range=(0, y_max), 
                            metrics = [rmse, rrmse, bias, bias_pct, mae, R2Score()])

        ensemble.fit_one_cycle(20, 1e-2)

        val_res = ensemble.validate()
        val_res.to_csv(f'{t}/ann_val.csv', float_format='%.4f')
        val_interp = RegressionInterpretation.from_ensemble(ensemble)

        fig = val_interp.plot_results()
        for f in fig: f.get_figure().savefig(f'{t}/ann_val.png', dpi=300, bbox_inches='tight')

        test_dls = test_tb.dataloaders(y_block=RegressionBlock(), shuffle_train=False, drop_last=False)
        test_res = ensemble.validate(dl=test_dls[0])
        test_res.to_csv(f'{t}/ann_test.csv', float_format='%.4f')
        test_interp =  RegressionInterpretation.from_ensemble(ensemble, dl=test_dls[0])
        
        fig = test_interp.plot_results()
        for f in fig: f.get_figure().savefig(f'{t}/ann_test.png', dpi=300, bbox_inches='tight')


        rf = RandomForestRegressor(n_estimators=500, max_features=0.5, min_samples_leaf=4, oob_score=True)
        rf.fit(trainval_tb.train.xs, trainval_tb.train.ys.values.ravel())
        fig = plot_sklearn_regression(rf, trainval_tb.valid.xs, trainval_tb.valid.ys)
        for f in fig: f.get_figure().savefig(f'{t}/rf_val.png', dpi=300, bbox_inches='tight')

        fig = plot_sklearn_regression(rf, test_tb.train.xs, test_tb-train.ys)
        for f in fig: f.get_figure().savefig(f'{t}/rf_test.png', dpi=300, bbox_inches='tight')




if __name__ == '__main__':
    # Todo argparse
    run_models()