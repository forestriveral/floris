import numpy as np
import pandas as pd
import originpro as op
from sklearn import linear_model

def bayesian(vX, vY):
    blr = linear_model.BayesianRidge(tol=1e-6, fit_intercept=True,
                                     compute_score=False, alpha_init=1,
                                     lambda_init=1e-3)
    blr.fit(np.vander(vX, 10), vY)
    mean = blr.predict(np.vander(vX, 10))
    return mean


def simple_draw():
    op.set_show()

    x_vals = [1,2,3,4,5,6,7,8,9,10]
    y_vals = [23,45,78,133,178,199,234,278,341,400]

    wks = op.new_sheet('w')

    wks.from_list(0, x_vals, 'X Values')
    wks.from_list(1, y_vals, 'Y Values')

    gp = op.new_graph()
    gl = gp[0]
    gl.add_plot(wks, 1, 0)
    gl.rescale()

    fpath = op.path('u') + 'simple.png'
    gp.save_fig(fpath)
    print(f'{gl} is exported as {fpath}')

    op.exit()


def data_reader_test():
    # get_data = np.loadtxt(f'./turb_30_4d_getdata.txt', comments='#')
    # print(get_data)

    ori_data = pd.read_csv(f'./turb_30_4d_origin.txt',)
    print(ori_data.columns)
    # ori_data = ori_data.dropna(axis=0, how='all')
    # print(ori_data)
    # exp_xy = [~np.isnan(a).any(axis=1)]
    exp_xy = ori_data[['exp_x', 'exp_y']].dropna(axis=0, how='all').values
    les1_xy = ori_data[['les1_x', 'les1_y']].dropna(axis=0, how='all').values
    print(exp_xy.shape)
    print(les1_xy.shape)


if __name__ == "__main__":
    # simple_draw()
    data_reader_test()