from dataset import get_dataset
from model import get_model
import torch
from config import get_gowalla_config
import numpy as np
import matplotlib.pyplot as plt
import dgl
import scipy.sparse as sp
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd


def main():
    """
    device = torch.device('cpu')
    config = get_gowalla_config(device)
    dataset_config, model_config, _ = config[2]
    dataset_config['path'] = dataset_config['path'][:-4] + str(1)
    model_config['name'] = 'AttIGCN'

    dataset = get_dataset(dataset_config)
    model = get_model(model_config, dataset)
    model.eval()
    model.load('checkpoints/AttIGCN_BPRTrainer_ProcessedDataset_12.221.pth')
    with torch.no_grad():
        alpha = model.inductive_rep_layer(model.feat_mat, return_alpha=True)
        row, column = model.feat_mat.indices()
        g = dgl.graph((row, column), num_nodes=max(model.feat_mat.shape), device=model.device)
        contribution = dgl.ops.gspmm(g, 'copy_rhs', 'sum', lhs_data=None, rhs_data=alpha)
        alpha = alpha.cpu().numpy()
        contribution = contribution[:-2].cpu().numpy()

    sub_mat = sp.coo_matrix((np.ones((len(dataset.train_array),)), np.array(dataset.train_array).T),
                            shape=(model.n_users, model.n_items), dtype=np.float32)
    user_degree = np.array(np.sum(sub_mat, axis=1)).squeeze()
    item_degree = np.array(np.sum(sub_mat, axis=0)).squeeze()
    ranked_users_degree = np.argsort(user_degree)[::-1].copy()
    ranked_items_degree = np.argsort(item_degree)[::-1].copy()
    selected_user = ranked_users_degree[2]
    selected_item = ranked_items_degree[2]
    user_alpha = alpha[row == selected_user]
    item_alpha = alpha[row == (model.n_users + selected_item)]
    print(user_alpha)
    print(item_alpha)

    pdf = PdfPages('figure_0.pdf')
    fig, ax = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(11, 3))
    axes = ax.flatten()
    eps = 5.e-5
    bins = np.arange(0., 1.e-2 + eps, eps)
    axes[0].hist(x=user_alpha, bins=bins, alpha=0.5)
    axes[0].set_xlim(0.5e-3, 3.5e-3)
    axes[0].ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    axes[0].set_xlabel('Weight', fontsize=17)
    axes[0].set_ylabel('Frequency', fontsize=17)
    axes[0].set_title('Attention weights on the interactions of a user', fontsize=17)
    axes[0].tick_params(labelsize=14)

    eps = 2.e-5
    bins = np.arange(0., 1.e-2 + eps, eps)
    axes[1].hist(x=item_alpha, bins=bins, alpha=0.5)
    axes[1].set_xlim(0., 2.e-3)
    axes[1].ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    axes[1].set_xlabel('Weight', fontsize=17)
    axes[1].set_ylabel('Frequency', fontsize=17)
    axes[1].set_title('Attention weights on the interactions of an item', fontsize=17)
    axes[1].tick_params(labelsize=14)
    pdf.savefig()
    plt.close(fig)
    pdf.close()

    pdf = PdfPages('figure_1.pdf')
    fig, ax = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(11, 3))
    axes = ax.flatten()
    eps = 0.1
    bins = np.arange(0., 10. + eps, eps)
    axes[0].hist(x=contribution[:model.n_users], bins=bins, alpha=0.5)
    axes[0].set_xlabel('Contribution', fontsize=17)
    axes[0].set_ylabel('Frequency', fontsize=17)
    axes[0].set_title('Global user contribution distribution', fontsize=17)
    axes[0].tick_params(labelsize=14)
    axes[0].set_xlim(0., 6.)

    eps = 0.05
    bins = np.arange(0., 5. + eps, eps)
    axes[1].hist(x=contribution[model.n_users:], bins=bins, alpha=0.5)
    axes[1].set_xlabel('Contribution', fontsize=17)
    axes[1].set_ylabel('Frequency', fontsize=17)
    axes[1].set_title('Global item contribution distribution', fontsize=17)
    axes[1].tick_params(labelsize=14)
    axes[1].set_xlim(0., 3.)
    pdf.savefig()
    plt.close(fig)
    pdf.close()
    """

    mf = [11.934] * 10
    imf_d = [9.094, 10.985, 11.953, 12.722, 13.261, 13.545, 13.63, 13.791, 13.994, 14.051]
    imf_nd = [9.415, 11.224, 12.288, 12.821, 13.263, 13.586, 13.806, 13.833, 14.036, 14.051]
    imf_pr = [9.438, 11.177, 12.202, 12.828, 13.19, 13.622, 13.821, 13.931, 13.946, 14.051]
    lgcn = [14.037] * 10
    igcn_d = [12.991, 14.1, 14.651, 14.973, 15.134, 15.298, 15.306, 15.344, 15.375, 15.42]
    igcn_nd = [13.259, 14.357, 14.917, 15.105, 15.269, 15.333, 15.451, 15.376, 15.379, 15.42]
    igcn_pr = [13.17, 14.173, 14.741, 15.111, 15.282, 15.368, 15.437, 15.427, 15.456, 15.42]
    ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
    pdf = PdfPages('figure_2.pdf')
    fig, ax = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(11, 4))
    axes = ax.flatten()
    axes[0].plot(ratio, np.array(mf) / 100., label='MF', marker='s')
    axes[0].plot(ratio, np.array(imf_d) / 100., label='INMO-MF-degree', marker='v')
    axes[0].plot(ratio, np.array(imf_nd) / 100., label='INMO-MF-normalized_degree', marker='o')
    axes[0].plot(ratio, np.array(imf_pr) / 100., label='INMO-MF-page_rank', marker='d')
    axes[0].set_xticks(ratio)
    axes[0].legend(fontsize=13)
    axes[0].set_xlabel('Percentage of core users and core items', fontsize=17)
    axes[0].set_ylabel('NDCG@20', fontsize=17)
    axes[0].set_title('INMO-MF', fontsize=17)
    axes[1].plot(ratio, np.array(lgcn) / 100., label='LightGCN', marker='s')
    axes[1].plot(ratio, np.array(igcn_d) / 100., label='INMO-LGCN-degree', marker='v')
    axes[1].plot(ratio, np.array(igcn_nd) / 100., label='INMO-LGCN-normalized_degree', marker='o')
    axes[1].plot(ratio, np.array(igcn_pr) / 100., label='INMO-LGCN-page_rank', marker='d')
    axes[1].set_xticks(ratio)
    axes[1].legend(fontsize=13)
    axes[1].set_xlabel('Percentage of core users and core items', fontsize=17)
    axes[1].set_ylabel('NDCG@20', fontsize=17)
    axes[1].set_title('INMO-LGCN', fontsize=17)
    pdf.savefig()
    plt.close(fig)
    pdf.close()

    imf = [13.848, 13.862, 13.906, 13.958, 13.751]
    igcn = [15.315, 15.378, 15.391, 15.172, 14.639]
    beta = ['0', '0.001', '0.01', '0.1', '1']
    pdf = PdfPages('figure_3.pdf')
    fig, ax = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(11, 4))
    axes = ax.flatten()
    axes[0].plot(np.array(imf) / 100., marker='s')
    axes[0].set_xticks([0, 1, 2, 3, 4])
    axes[0].set_xticklabels(beta)
    axes[0].set_xlabel('Weight of self-enhanced loss', fontsize=17)
    axes[0].set_ylabel('NDCG@20', fontsize=17)
    axes[0].set_title('INMO-MF', fontsize=17)
    axes[1].plot(np.array(igcn) / 100., marker='s')
    axes[1].set_xticks([0, 1, 2, 3, 4])
    axes[1].set_xticklabels(beta)
    axes[1].set_xlabel('Weight of self-enhanced loss', fontsize=17)
    axes[1].set_ylabel('NDCG@20', fontsize=17)
    axes[1].set_title('INMO-LGCN', fontsize=17)
    pdf.savefig()
    plt.close(fig)
    pdf.close()

    def read_csv_log(path):
        df = pd.read_csv(path)
        x = df['Step'].values
        y = df['Value'].values
        return x, y

    pdf = PdfPages('figure_4.pdf')
    fig, ax = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(11, 4))
    axes = ax.flatten()
    x, y = read_csv_log('/Users/wuyunfan/work/papers/21-3-code/igcn/logs/final/imf/gowalla/1/csv.csv')
    axes[0].plot(x, y, label='INMO-MF')
    x, y = read_csv_log('/Users/wuyunfan/work/papers/21-3-code/igcn/logs/ablation/anneal/imf/csv.csv')
    axes[0].plot(x, y, label='INMO-MF w/o NA')
    axes[0].set_xlabel('Epoch', fontsize=17)
    axes[0].set_ylabel('NDCG@20', fontsize=17)
    axes[0].set_title('INMO-MF', fontsize=17)
    axes[0].legend(fontsize=14, loc=4)
    x, y = read_csv_log('/Users/wuyunfan/work/papers/21-3-code/igcn/logs/final/igcn/gowalla/1/csv.csv')
    axes[1].plot(x, y, label='INMO-LGCN')
    x, y = read_csv_log('/Users/wuyunfan/work/papers/21-3-code/igcn/logs/ablation/anneal/igcn/csv.csv')
    axes[1].plot(x, y, label='INMO-LGCN w/o NA')
    axes[1].set_xlabel('Epoch', fontsize=17)
    axes[1].set_ylabel('NDCG@20', fontsize=17)
    axes[1].set_title('INMO-LGCN', fontsize=17)
    axes[1].legend(fontsize=14, loc=4)
    pdf.savefig()
    plt.close(fig)
    pdf.close()


if __name__ == '__main__':
    main()
