import numpy as np
import pandas as pd
import originpro as op
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
    print(np.all(~np.isnan(exp_xy)))
    assert np.all(~np.isnan(exp_xy)) and np.all(~np.isnan(les1_xy))
    print(exp_xy.shape)
    print(les1_xy.shape)


def sample_test(id=0):
    # Ensures that the Origin instance gets shut down properly.
    import sys
    def origin_shutdown_exception_hook(exctype, value, traceback):
        op.exit()
        sys.__excepthook__(exctype, value, traceback)
    if op and op.oext:
        sys.excepthook = origin_shutdown_exception_hook


    # Set Origin instance visibility.
    if op.oext:
        op.set_show(True)

    # YOUR CODE HERE
    if id == 0:

        # Example of opening a project and reading data.
        # We'll open the Tutorial Data.opju project that ships with Origin.
        src_opju = op.path('e') + r'Samples\Tutorial Data.opju'
        op.open(file = src_opju, readonly = True)
        # Simple syntax to find a worksheet.
        src_wks = op.find_sheet('w', 'Book1A')
        # Pull first column data into a list and dump to screen.
        lst = src_wks.to_list(0)
        print(*lst, sep = ", ")
        # Pull entire worksheet into a pandas DataFrame and partially dump to screen.
        # Column long name will be columns name.
        df = src_wks.to_df()
        print(df.head())
        # Start a new project which closes the previous one.
        op.new()
        #Examples of writing data to a project and saving it.
        # We'll reuse the data objects we previously created.
        # Simple syntax to create a new workbook with one sheet
        dest_wks = op.new_sheet('w')
        # Insert list data into columns 1
        dest_wks.from_list(0, lst)
        # Add another sheet top above workbook and add the DataFrame data.
        # DataFrame column names with be Origin column Long Names.
        dest_wks.get_book().add_sheet().from_df(df)
        # Save the opju to your UFF.
        op.save(op.path('u')+ 'Ext Python Example 1.opju')

    # Exit running instance of Origin.
    if op.oext:
        op.exit()


def double_gaussian():

    def test_func(r, *args):
        a, b, sigma = args[0], args[1], args[2]
        return 1 / (a * r + b) * np.exp(- r**2 / (2 * sigma**2))

    a_list, b_list, simga_list = [0.1, ], [2., ], [1., 2.]
    colors = list(mcolors.TABLEAU_COLORS.keys())
    x = np.arange(-10, 10, 0.5)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
    for i, a_i in enumerate(a_list):
        for j, b_j in enumerate(b_list):
            for k, sigma_k in enumerate(simga_list):
                ind = (i + 1) * (j + 1) * (k + 1)
                ax.plot(x, test_func(x, a_i, b_j, sigma_k),
                        c=colors[ind], lw=2., ls='-',
                        label=f'a={a_i},b={b_j},sigma={sigma_k}')
    ax.set_xlim([-6., 6.])
    ax.legend()
    plt.show()



if __name__ == "__main__":
    # simple_draw()
    # data_reader_test()
    # sample_test(id=0)
    double_gaussian()