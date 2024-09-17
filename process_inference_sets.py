import pandas as pd
import numpy as np
from pathlib import Path
import scipy as sp
from matplotlib import pyplot as plt
import seaborn as sns

def merge_counts(gt_df, pd_df):
    assert len(gt_df) == len(pd_df), f"Length of the two dataframes does not match."
    gt_df.rename(columns={"num_stomata": "gt"}, inplace=True)
    pd_df.rename(columns={"filtered_num_stomata": "pred"}, inplace=True)
    count_result = gt_df.merge(pd_df, on=["Sample", "Image"])
    return count_result


def calculate_density(merged_df, factor=1.789):
    merged_df['gt_density'] = merged_df['gt'] * factor
    merged_df['pd_density'] = merged_df['pred'] * factor
    return

def plot_correlation_counts(merged_df, save_path, lims, text):
    pal = ["#C2CD23", "#918BC3"]
    plt.rcParams.update({'mathtext.default': 'regular'})
    sns.set_palette(palette=pal)
    plt.xlim(lims)
    plt.ylim(lims)
    ax = plt.gca()
    ax.grid(False)
    ax.text(80, 100, f'{text}')
    sns.scatterplot(x="gt_density", y="pd_density", data=merged_df, hue="Inference Set")
    h, l = ax.get_legend_handles_labels()
    ax.legend(labels=["NILs", "$\it{ZmAbh}$ mutants"], handles = h)
    sns.regplot(data=merged_df, x="gt_density", y="pd_density", scatter=False, ax=ax, color="black")
    save_path.mkdir(exist_ok=True, parents=True)
    ax.set_xlabel('SD by manual scoring $[mm^{-2}]$')
    ax.set_ylabel('SD predicted by our model $[mm^{-2}]$')
    plt.savefig(f"{str(save_path)}/figure_correlation_plot.pdf", dpi=600, bbox_inches='tight')
    plt.show()
    return

def extract_from_file_path(df):
    df[['zip_folder', 'Sample', 'Image']] = df['file_path'].str.split('/', n=2, expand=True)
    df[['Image', 'suffix']] = df['Image'].str.split('.', n=2, expand=True)
    df = df[["Sample", "Image", "filtered_num_stomata"]].copy()
    df['Sample'] = df['Sample'].astype(str)
    return df

def group_by_sample(gt_df, pd_df):
    df1 = gt_df.groupby('Sample')["gt"].mean().reset_index()
    df2 = pd_df.groupby('Sample')["pred"].mean().reset_index()
    df1['Sample'] = df1['Sample'].astype(str)
    count_result = df1.merge(df2, on="Sample")
    return df1, df2, count_result


def process_data(gt_path, pred_path, inference_set):
    pd_csv_path = Path(f"{str(pred_path)}/summary.csv")
    pds = pd.read_csv(pd_csv_path, dtype={'unfiltered_num_stomata': np.uint16, 'filtered_num_stomata':np.uint16})
    pds = pds.loc[pds.file_path.str.contains(f"{inference_set}")]
    pds = extract_from_file_path(pds)
    gt_csv_path = Path(f"{str(gt_path)}/{inference_set}_gt.csv")
    gts = pd.read_csv(gt_csv_path)
    gts = gts[["Sample", "1","2","3","4","5","6","7","8","9"]]
    gts = gts.melt(id_vars=["Sample"], var_name="Image", value_name="num_stomata")
    gts.dropna(inplace=True)
    gts = gts.loc[gts.num_stomata!=0]
    gts['Sample'] = gts['Sample'].astype(str)
    merged_df = merge_counts(gts, pds)
    print(f"Processing {len(merged_df)} images...")
    calculate_density(merged_df)
    gt_df, pred_df, df1= group_by_sample(gts, pds)
    calculate_density(df1)
    genotypes = df1.merge(gts[["Sample"]], on="Sample")
    genotypes["delta"] = np.abs(genotypes["gt"] - genotypes["pred"])
    return genotypes, merged_df, gts


if __name__ == "__main__":
    save_path = Path("./output")
    pred_path = Path("./data/predictions")
    gt_path = Path("./data/inference")
    genotypes_heat, merged_df_heat, gts_heat = process_data(gt_path, pred_path, inference_set="Heat_L6")
    genotypes_mut, merged_df_mut, gts_mut = process_data(gt_path, pred_path, inference_set="Inference_mutants")
    genotypes_mut["Inference Set"] = "ZmAbh mutants"
    genotypes_heat["Inference Set"] = "NILs"
    genotypes = pd.concat([genotypes_heat, genotypes_mut])
    merged_df = pd.concat([merged_df_mut, merged_df_heat])
    r, p = sp.stats.pearsonr(x=genotypes['gt_density'], y=genotypes['pd_density'])
    plot_correlation_counts(genotypes, save_path, lims=[54, 106], text=f"r={r:0.3f}")
    
    print(f"Pearson R: {r} with p {p}")
