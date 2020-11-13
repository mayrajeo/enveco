from fastai.data.all import *
from fastai.vision.all import *
absolute_path = Path().absolute()
sys.path.append(str(absolute_path.parents[0]))

from enveco.data.las import *
from enveco.models.inception3dv3 import *
from enveco.interpretation import *
from enveco.metrics import *

import argparse

def run_models(basedir, train_csv, valid_csv, test_csv, outdir):
    if not os.path.exists(outdir): os.makedirs(outdir)

    basedir = Path(basedir)

    train_df = pd.read_csv(basedir/train_csv)
    valid_df = pd.read_csv(basedir/valid_csv)
    train_df['is_val'] = False
    valid_df['is_val'] = True
    test_df = pd.read_csv(basedir/test_csv)
    df = pd.concat((train_df, valid_df))
    targets = ['v', 'd', 'h', 'g']
    for t in targets:
        # No voxel processing
        if not os.path.exists(f'{outdir}/{t}'): os.makedirs(f'{outdir}/{t}')
        dls = VoxelDataLoaders.from_df(df, path=basedir, folder='AV_las', 
                                       bin_voxels=True,
                                       bottom_voxels=False,
                                       mask_plot=True,
                                       y_block=RegressionBlock(),
                                       label_col=t,
                                       fn_col='sampleplitid', bs=32
                                       batch_tfms=[DihedralItem])

        y_max = train_df[t].max() * 1.1

        metrics = [rmse, rrmse, bias, bias_pct, mae, R2Score()]
        ens = Ensemble(dls, path='.', metrics=metrics, y_range=(0, y_max), 
                       learn_func=inception_learner)

        ens.fit_one_cycle(20, max_lr=1e-2)

        ens_res = ens.validate()
        ens_res.to_csv(f'{outdir}/{t}/inception_val.csv', float_format='%.4f')

        ens_interp = RegressionInterpretation.from_ensemble(ens)
        fig = ens_interp.plot_results()
        for f in fig: f.get_figure().savefig(f'{outdir}/{t}/inception_val.png', dpi=300, bbox_inches='tight')

        test_dl = inc_learner.dls.test_dl(test_df, with_labels=True)
        test_res = ens.validate(dl=test_dl)
        test_res.to_csv(f'{outdir}/{t}/inception_test.csv', float_foramat='%.4f')

        test_interp = RegressionInterpretation.from_ensemble(ens, dl=test_dl)
        fig = test_interp.plot_results()
        for f in fig: f.get_figure().savefig(f'{outdir}/{t}/inception_test.png', dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    run_3dcnn_models()