import pandas as pd
from sklearn import preprocessing
from sklearn import metrics
import numpy as np 

# create function for preparing data:
def prep_data_all_epochs(results: pd.DataFrame, 
              results_val:pd.DataFrame, 
              results_4500:pd.DataFrame, 
              min_epoch:int = 20):
    
    # Filter data to start with 10 epochs:
    # anything below a certain number of epochs seems to be very flaky.
    #min_epoch = 20
    mask = results.epochs_trained >= min_epoch
    results = results[mask].reset_index(drop=True)
    mask = results_val.epochs_trained >= min_epoch
    results_val = results_val[mask].reset_index(drop=True)
    mask = results_4500.epochs_trained >= min_epoch
    results_4500 = results_4500[mask].reset_index(drop=True)
    classes = results.columns.tolist()[2:12]
    print(classes)
    results["total_training_size"] = results[classes].sum(axis=1)
    results_val["total_training_size"] = results_val[classes].sum(axis=1)
    results.head(4)
    results_4500["total_training_size"] = results_4500[classes].sum(axis=1)
    results_4500.head(4)
    print(results[["accs"] + classes + ["epochs_trained", "total_training_size"]].iloc[:5, :].to_latex(float_format="{:.2f}".format))


    ### constructing a pred dataset: 
    results_pred = results[["training_times"] + classes + ["epochs_trained", "total_training_size"]].copy()
    row = {c: 5000 for c in classes}
    row["epochs_trained"] = 195
    row["total_training_size"] =  50000
    row["training_times"] = 100
    results_pred = results_pred.append(row, ignore_index=True)
    row = {c: 5000 for c in classes}
    row["training_times"] = 100
    row["epochs_trained"] = 150
    row["total_training_size"] =  50000
    results_pred = results_pred.append(row, ignore_index=True)
    results_pred = results_pred.iloc[-2:,:].reset_index(drop=True)
    for c_5000 in classes:
        row = {c: 4500 for c in classes}
        row[c_5000] = 5000
        row["epochs_trained"] = 195
        row["training_times"] = 100
        row["total_training_size"] =  45500
        results_pred = results_pred.append(row, ignore_index=True)
        row["epochs_trained"] = 150
        results_pred = results_pred.append(row, ignore_index=True)
        row["epochs_trained"] = 100
        results_pred = results_pred.append(row, ignore_index=True)
        row["epochs_trained"] = 50
        results_pred = results_pred.append(row, ignore_index=True)
    for c_5000 in classes:
        row = {c: 4000 for c in classes}
        row[c_5000] = 5000
        row["epochs_trained"] = 195
        row["training_times"] = 100
        row["total_training_size"] =  41000
        results_pred = results_pred.append(row, ignore_index=True)
        row["epochs_trained"] = 150
        results_pred = results_pred.append(row, ignore_index=True)
        row["epochs_trained"] = 100
        results_pred = results_pred.append(row, ignore_index=True)
        row["epochs_trained"] = 50
        results_pred = results_pred.append(row, ignore_index=True)
    for c_5000 in classes:
        row = {c: 5000 for c in classes}
        row[c_5000] = 7000
        row["epochs_trained"] = 195
        row["training_times"] = 100
        row["total_training_size"] =  52000
        results_pred = results_pred.append(row, ignore_index=True)
        row["epochs_trained"] = 150
        results_pred = results_pred.append(row, ignore_index=True)
        row["epochs_trained"] = 100
        results_pred = results_pred.append(row, ignore_index=True)
        row["epochs_trained"] = 50
        results_pred = results_pred.append(row, ignore_index=True)

    results_pred_orig = results_pred.copy()

    ### normalize results for train:
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(X = results.iloc[:, 1:])
    results_scaled = pd.DataFrame(scaler.transform(results.iloc[:, 1:]))
    results_scaled.columns = results.iloc[:, 1:].columns
    results = pd.concat([results["accs"], results_scaled], axis=1)
    results.head(2)

    # using the same scaler for val:
    results_scaled_val = pd.DataFrame(scaler.transform(results_val.iloc[:, 1:]))
    results_scaled_val.columns = results_val.iloc[:, 1:].columns
    results_val = pd.concat([results_val["accs"], results_scaled_val], axis=1)
    results_val.head(2)

    # using the same scaler for pred tries:
    results_scaled_pred = pd.DataFrame(scaler.transform(results_pred))
    results_scaled_pred.columns = results_pred.columns
    results_pred = results_scaled_pred
    results_pred
    
    results_4500_orig = results_4500.copy()
    results_scaled_4500 = pd.DataFrame(scaler.transform(results_4500.iloc[:, 2:]))
    results_scaled_4500.columns = results_4500.iloc[:, 2:].columns
    #results_4500 = results_4500_scaled
    #results_4500.head(3)
    results_4500 = pd.concat([results_4500["accs"], results_scaled_4500], axis=1)
    results_4500.head(2)


    results_4500["class"] = results_4500[classes].idxmax(axis=1)
    results_4500.head(2)

    # prep data for using class counts:
    #xdata = np.transpose(results.to_numpy()[:, 2:-1])
    xdata = np.transpose(results.to_numpy()[:, 2:])
    y = results.to_numpy()[:, 0]
    #xdata_val = np.transpose(results_val.to_numpy()[:, 2:-1])
    xdata_val = np.transpose(results_val.to_numpy()[:, 2:])
    y_val = results_val.to_numpy()[:, 0]
    xdata.shape
    #xdata_pred = np.transpose(results_pred.to_numpy()[:, 1:-1])
    xdata_pred = np.transpose(results_pred.to_numpy()[:, 1:])
    xdata_pred.shape

    #display(results_4500.head(2))
    display(results_4500_orig.head())
    #xdata_4500 = np.transpose(results_4500.loc[:, classes + ["epochs_trained"] ].to_numpy())
    xdata_4500 = np.transpose(results_4500.loc[:, classes + ["epochs_trained", "total_training_size"] ].to_numpy())
    print(xdata_4500.shape)
    #xdata_4500
    results_4500.head(2)
    y_4500 = results_4500.to_numpy()[:, 0]
    display(results_4500_orig.head())

    # data for last_epoch_training:
    mask = results.epochs_trained == 1
    results_last_epoch = results[mask].reset_index(drop=True)
    xdata_last_epoch = np.transpose(results_last_epoch.to_numpy()[:, 2:-2])
    y_last_epoch = results_last_epoch.to_numpy()[:, 0]

    mask = results_val.epochs_trained == 1
    results_val_last_epoch = results_val[mask].reset_index(drop=True)
    xdata_val_last_epoch = np.transpose(results_val_last_epoch.to_numpy()[:, 2:-2])
    y_val_last_epoch = results_val_last_epoch.to_numpy()[:, 0]
    
    # data for total_n and epoch only:
    xdata_total_n_epoch = np.transpose(results.to_numpy()[:, -2:])
    xdata_val_total_n_epoch = np.transpose(results_val.to_numpy()[:, -2:])
    xpred_total_n_epoch = np.transpose(results_pred.to_numpy()[:, -2:])
    xpred_total_n_epoch.shape

    # data for last epoch, total_n only:
    xdata_last_epoch_total_n = np.transpose(results_last_epoch.to_numpy()[:, -1:])
    xdata_val_last_epoch_total_n = np.transpose(results_val_last_epoch.to_numpy()[:, -1:])

    results = {"classes": classes, 
               "xdata": xdata, 
               "y": y, 
               "xdata_val": xdata_val, 
               "y_val": y_val, 
               "xdata_pred": xdata_pred, 
               "xdata_4500": xdata_4500, 
               "y_4500": y_4500, 
               "results": results, 
               "results_val": results_val, 
               "results_4500": results_4500, 
               "results_pred": results_pred, 
               "results_4500_orig": results_4500_orig, 
               "results_pred_orig": results_pred_orig, 
               "xdata_last_epoch": xdata_last_epoch, 
               "y_last_epoch": y_last_epoch, 
               "xdata_val_last_epoch": xdata_val_last_epoch, 
               "y_val_last_epoch": y_val_last_epoch, 
               "xdata_total_n_epoch": xdata_total_n_epoch, 
               "xdata_val_total_n_epoch": xdata_val_total_n_epoch}
    
    
    return results

# next todo: write function for create last eoch only versions.