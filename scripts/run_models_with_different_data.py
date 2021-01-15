from fastai.data.all import *
from fastai.tabular.all import *
absolute_path = Path().absolute()
sys.path.append(str(absolute_path.parents[0]))

from enveco.tabular.preprocessing import *
from enveco.model.ensemble import *
from enveco.interpretation import *
from enveco.metrics import *

from sklearn.ensemble import RandomForestRegressor

import argparse
from itertools import product

def run_models(data_path:str, outdir:str, preprocess=False):
    target_variables = ['v', 'g', 'h', 'd', 'v_ma', 'v_ku', 'v_lp']

    data_path = Path(data_path)
    if not os.path.exists(outdir): os.makedirs(outdir)

    preprocessor = EnvecoPreprocessor(data_path/'AV.leaf.on.train.csv',
                                      data_path/'AV.leaf.on.val.csv',
                                      data_path/'AV.leaf.on.test.csv')

    if preprocess == True: 
        trainval_tb, test_tb = preprocessor.preprocess_lidar(target_col=['v_lp'], 
                                                    path=data_path/'AV_las',
                                                    mask_plot=True,
                                                    min_h=1.3,
                                                    normalize=True,
                                                    log_y=False,
                                                    save_path='model_data')

        # Preprocess image data
        trainval_tb, test_tb = preprocessor.preprocess_image(target_col=['v_lp'], 
                                                    path=data_path/'AV_tif',
                                                    mask_plot=True,
                                                    normalize=True,
                                                    log_y=False,
                                                    save_path='model_data')

        # Preprocess both
        trainval_tb, test_tb = preprocessor.preprocess(target_col=['v_lp'], 
                                                    path=data_path,
                                                    lidar_pref='AV_las',
                                                    image_pref='AV_tif',
                                                    mask_plot=False,
                                                    min_h=1.3,
                                                    normalize=True,
                                                    log_y=False,
                                                    save_path='model_data')

    feature_space = ['las', 'image', 'both']
    log_tfm = [True, False]

    for target, feature, log_y in product(target_variables, feature_space, log_tfm):
        if feature == 'las':
            trainval_tb, test_tb = preprocessor.load_las(path='model_data', 
                                                         target_col=target, 
                                                         log_y=log_y)
        if feature == 'image':
            trainval_tb, test_tb = preprocessor.load_image(path='model_data',
                                                           target_col=target,
                                                           log_y=log_y)
        if feature == 'both':
            trainval_tb, test_tb= preprocessor.load_las_image(path='model_data',
                                                              target_col=target,
                                                              log_y=log_y)
        savedir = f'{target}_{feature}{"_log" if log_y else ""}'
        if not os.path.exists(f'{outdir}/{savedir}'): os.makedirs(f'{outdir}/{savedir}')

        dls = trainval_tb.dataloaders(bs=64, y_block=RegressionBlock())

        y_max = trainval_tb.train.y.max() * 1.1 
        print(f'Running models for {target}, using {feature} features, with y_max of {y_max} and log_transform {log_y}')
        ensemble = Ensemble(dls, learn_func=tabular_learner,
                            y_range=(0, y_max), 
                            metrics = [rmse, rrmse, bias, bias_pct, mae, R2Score()],
                            n_models=10)

        ensemble.fit_one_cycle(20, 1e-2)

        val_res = ensemble.validate()
        val_res.to_csv(f'{outdir}/{savedir}/ann_val.csv', float_format='%.4f', index=False)
        val_interp = RegressionInterpretation.from_ensemble(ensemble)

        fig = val_interp.plot_results(log_y=log_y)
        for f in fig: f.get_figure().savefig(f'{outdir}/{savedir}/ann_val.png', dpi=300, bbox_inches='tight')

        test_dls = test_tb.dataloaders(y_block=RegressionBlock(), shuffle_train=False, drop_last=False)
        test_res = ensemble.validate(dl=test_dls[0])
        test_res.to_csv(f'{outdir}/{savedir}/ann_test.csv', float_format='%.4f', index=False)
        test_interp =  RegressionInterpretation.from_ensemble(ensemble, dl=test_dls[0])
        
        fig = test_interp.plot_results(log_y=log_y)
        for f in fig: f.get_figure().savefig(f'{outdir}/{savedir}/ann_test.png', dpi=300, bbox_inches='tight')

        ensemble.export(folder=f'{outdir}/{savedir}/models')

        rf = RandomForestRegressor(n_estimators=500, max_features=0.5, min_samples_leaf=4, oob_score=True)
        rf.fit(trainval_tb.train.xs, trainval_tb.train.ys.values.ravel())
        fig = plot_sklearn_regression(rf, trainval_tb.valid.xs, trainval_tb.valid.ys,log_y=log_y)
        for f in fig: f.get_figure().savefig(f'{outdir}/{savedir}/rf_val.png', dpi=300, bbox_inches='tight')

        fig = plot_sklearn_regression(rf, test_tb.train.xs, test_tb.train.ys, log_y=log_y)
        for f in fig: f.get_figure().savefig(f'{outdir}/{savedir}/rf_test.png', dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    # Todo argparse
    run_models('../../enveco_data/enveco/', 'results_15_1', False)