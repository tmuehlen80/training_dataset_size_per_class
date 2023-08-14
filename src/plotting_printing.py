
import pandas as pd
from sklearn import preprocessing
from sklearn import metrics
import numpy as np 

import pandas as pd
from IPython.display import HTML
from IPython.display import display
import numpy as np
import warnings
from pandas.core.common import SettingWithCopyWarning
import matplotlib.pyplot as plt
import seaborn as sns
import os

from scipy.optimize import curve_fit

from sklearn import preprocessing
from sklearn import metrics


def plotting_printing_all_epochs(func, model_type, y_hat_name, xdata, y, param_names,  params, results, xdata_val, y_val, results_val, xdata_4500, xdata_pred, results_pred_orig, results_4500, results_4500_orig, saving_plots=False, is_2param=False):
    # some plotting parameters
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    param_df = pd.DataFrame({"param_name": param_names, "param_value": params})
    param_df = param_df.set_index("param_name").T
    HTML(display(param_df))
    print(param_df.to_latex(float_format="{:.2f}".format))
    y_hat = func(xdata, *params)
    print(((y_hat - y)**2).mean())
    # dataframe for plotting:
    results[y_hat_name] = y_hat

    sns.scatterplot(data = results, x="accs", y = y_hat_name, hue = "epochs_trained")
    plt.xlim((0.05, 0.9))
    plt.ylim((0.05, 0.9))
    plt.axline((0, 0), slope=1, color="black", linestyle=(0, (5, 5)))
    plt.text(x=0.1, y=0.8, s=f"r-sq: {np.round(metrics.r2_score(results.accs, results[y_hat_name]), 3)}", bbox=props)
    if saving_plots:
        plt.savefig(f"paper/plots/{model_type}_{y_hat_name}.jpg")
    plt.show()
    print(metrics.r2_score(results.accs, results[y_hat_name]))
    # val data:
    y_val_hat = func(xdata_val, *params)
    print(f"mean val loss: {((y_val_hat - y_val)**2).mean()}")
    # dataframe for plotting:
    results_val[y_hat_name] = y_val_hat

    sns.scatterplot(data = results_val, x="accs", y = y_hat_name, hue = "epochs_trained")
    plt.xlim((0.05, 0.9))
    plt.ylim((0.05, 0.9))
    plt.axline((0, 0), slope=1, color="black", linestyle=(0, (5, 5)))
    plt.text(x=0.1, y=0.8, s=f"r-sq: {np.round(metrics.r2_score(results_val.accs, results_val[y_hat_name]), 3)}", bbox=props)
    if saving_plots:
        plt.savefig(f"paper/plots/{model_type}_{y_hat_name}_val.jpg")
    plt.show()

    print(f"r_square val: {metrics.r2_score(results_val.accs, results_val[y_hat_name])}")
    
    acc_pred = func(xdata_pred, *params)
    results_pred_orig["acc_pred"] = acc_pred
    sns.scatterplot(data = results_pred_orig, x = "total_training_size", y = "acc_pred", hue="epochs_trained")
    plt.show()
    display(results_pred_orig)


    acc_4500 = func(xdata_4500, *params)
    results_4500[y_hat_name] = acc_4500
    results_4500_orig[y_hat_name] = acc_4500

    sns.scatterplot(data = results_4500_orig, x="accs", y = y_hat_name, hue = "epochs_trained")
    plt.xlim((0.7, 0.9))
    plt.ylim((0.7, 0.9))
    if saving_plots:
        plt.savefig(f"paper/plots/{model_type}_{y_hat_name}_4500.jpg")
    plt.show()

    results_4500.head(2)

    mask = results_4500_orig.epochs_trained == 195
    plt.scatter(x=results_4500[mask]["class"], y = results_4500[mask]["accs"], label= "accs")
    plt.scatter(x=results_4500[mask]["class"], y = results_4500[mask][y_hat_name], label= "accs pred")
    if saving_plots:
        plt.savefig(f"paper/plots/{model_type}_{y_hat_name}_4500_last_epoch.jpg")
    plt.show()
    results_4500_orig.head(2)
    
    if is_2param:
        param_unstacked = param_df.T.reset_index(drop=False).iloc[3:,:].param_name.str.split("_", expand=True).reset_index(drop=True)
        param_unstacked.columns = ["feature", "param_no"]
        param_unstacked["param_value"] = param_df.T.reset_index(drop=False).iloc[3:,:].reset_index(drop=True).param_value
        param_unstacked = param_unstacked.set_index(["feature", "param_no"])
        param_unstacked = param_unstacked.unstack()
        display(param_unstacked)

        param_unstacked.iloc[:,1]
        plt.scatter(param_unstacked.iloc[:,0], param_unstacked.iloc[:,1])
        plt.xlabel("parameter height")
        plt.ylabel("parameter width")
        for i in range(param_unstacked.shape[0]):
            plt.annotate(param_unstacked.reset_index().iloc[i,0], (param_unstacked.iloc[i,0] + 0.01, param_unstacked.iloc[i, 1]))
        if saving_plots:
            plt.savefig(f"paper/plots/{model_type}_{y_hat_name}_param1_vs_param2.jpg")
        plt.show()


