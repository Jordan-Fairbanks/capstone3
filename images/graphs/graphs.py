import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import psycopg2
import glob
from PIL import Image
from sklearn.manifold import TSNE

def plot_result_with_std(ax, labels, results, stds,xlabel='', **kwargs):
    """
    takes an axis objext, a list of labels, a results vecotor, and a standard deviation
    vector and returns a horizontal bar plot  
    """
    if kwargs:
        ax.barh(range(len(labels)), results, xerr=stds, **kwargs)
        ax.set_xlabel(xlabel)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
    else:
        ax.barh(range(len(labels)), results, xerr=stds)
        ax.set_xlabel(xlabel)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.invert_yaxis()

    return

def plot_tsne(ax, table, n_dim=2):
    """
    preforms the tsne algorithm on a set of predictions stored in a sql table
    and plots the result in a corresponding axes object. can be either 2D or 3D
    depending on n_dim (default=2)
    """
    cur.execute(f"SELECT * FROM {table}")
    info = np.array(cur.fetchall())[:,1:]
    tsne = TSNE(n_components=n_dim)
    reduced = tsne.fit_transform(info)
    if n_dim == 2:
        ax.scatter(reduced[:,0],reduced[:,1], alpha=.3, color='black')
    elif n_dim == 3:
        ax.scatter3D(reduced[:,0],reduced[:,1],reduced[:,2], alpha=.2, color='black')
    return

def make_3d_animation(table, title, path):
    """
    generates graphs on a 3d plot  to make a rotating gif and saves the figures in 
    a specified directory with a relative path
    """
    cur.execute(f"SELECT * FROM {table}")
    info = np.array(cur.fetchall())[:,1:]
    tsne = TSNE(n_components=3)
    reduced = tsne.fit_transform(info)
    angles = np.arange(0,361,10)
    for angle in angles:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.view_init(40, angle)
        ax.scatter3D(reduced[:,0], reduced[:,1], reduced[:,2], color='black', alpha=.2)
        ax.set_title(title)
        plt.savefig(f"{path}tsne_3d_angle{angle}.JPG")
        print(f"angle {angle} saved...")
    return 

def make_gif(frame_folder, path):
    frames = [Image.open(image) for image in glob.glob(f"{frame_folder}/*.JPG")]
    frame_one = frames[0]
    frame_one.save(f"{path}my_awesome.gif", format="GIF", append_images=frames,
               save_all=True, duration=100, loop=0)

if __name__ == "__main__":

    conn = psycopg2.connect(dbname='Food', user='postgres', password='galvanize', host='localhost', port='5432')
    cur = conn.cursor()

    model_names = ["SVD","NMF","KNNBasic","KNNWithMeans","Co-Clustering","Slope One"]
    colors = ["b","r","b","b","b","b"]
    rmses = np.array([0.8081,0.8560,0.8934,0.8799,0.9215,0.9016])
    rmse_std = np.array([0.0486,0.0208,0.0352,0.0265,0.0376,0.0274])
    maes = np.array([0.4850,0.5373,0.5153,0.5115,0.5183,0.5383])
    maes_std = np.array([0.0150,0.0058,0.0123,0.0075,0.0143,0.0121])
    fit_times = np.array([1.87,2.87,0.09,0.12,2.50,2.24])
    fit_std = np.array([0.14,0.08,0.01,0.,0.37,0.25])
    test_times = np.array([0.12,0.06,0.17,0.17,0.08,0.17])
    test_std = np.array([0.12,0.01,0.08,0.07,0.07,0.03])

    fig, ax = plt.subplots(figsize=(5,3))
    plot_result_with_std(ax, model_names, rmses, rmse_std, xlabel='Error', color=['r','r','b','b','b','b'])
    ax.axvline(rmses.mean(), linestyle='--', color='green', alpha=.6, label='Average')
    ax.legend()
    ax.set_title('RMSE')
    fig.tight_layout()
    plt.savefig('RMSE.png')

    fig, ax = plt.subplots(figsize=(5,3))
    plot_result_with_std(ax, model_names, maes, maes_std,xlabel='Error', color='blue')
    ax.set_title('MAE')
    fig.tight_layout()
    plt.savefig('MAE.png')

    fig, ax = plt.subplots(figsize=(5,3))
    plot_result_with_std(ax, model_names, fit_times, fit_std, xlabel='Time (seconds)', color=colors)
    ax.set_title('Time to Train Algorithm')
    fig.tight_layout()
    plt.savefig('fit_times.png')

    fig, ax = plt.subplots(figsize=(5,3))
    plot_result_with_std(ax, model_names, test_times, test_std, xlabel='Time (seconds)', color=colors)
    ax.set_title('Time to Make Reccomendation')
    fig.tight_layout()
    plt.savefig('test_times.png')
    print("BAR GRAPHS GENERATED...")

    # fig, ax = plt.subplots(figsize=(5,5))
    # plot_tsne(ax,'nmf_predictions')
    # ax.set_title('NMF Clusters')
    # plt.savefig('NMF_tsne.png')

    # fig, ax = plt.subplots(figsize=(5,5))
    # plot_tsne(ax,'knnb_predictions')
    # ax.set_title('KNNBasic')
    # plt.savefig('KNNB_tsne.png')
    
    # fig, ax = plt.subplots(figsize=(5,5))
    # plot_tsne(ax,'coclust_predictions')
    # ax.set_title('CoClustering')
    # plt.savefig('coclust_tsne.png')

    # print("TSNE 2D DONE...")

    # # fig = plt.figure()
    # # ax = plt.axes(projection="3d")
    # # plot_tsne(ax, 'svd_predictions', n_dim=3)
    # # ax.set_title('SVD Clusters')
    # # plt.savefig('svd_tsne3d.png')

    # # print("TSNE 3D DONE...")

    # make_3d_animation('svd_predictions','SVD Clusters', 'animation/')
    # make_gif('animation', 'animation/')
