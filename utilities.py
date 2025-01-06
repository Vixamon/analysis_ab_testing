"""
This module provides utility functions for exploratory data analysis,
statistical testing, and data visualization for A/B testing experiments and
other datasets.

Includes functions to find missing values, detect outliers, generate visualizations,
calculate confidence intervals, and perform statistical tests.
"""
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chisquare, binomtest
from statsmodels.stats.proportion import proportions_ztest

def find_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies and counts missing values in a DataFrame, including zeroes, empty strings, and NaN values.

    Args:
        df: The DataFrame to analyze.

    Returns:
        A DataFrame with counts of zeroes, empty strings, and NaN values for each column.
    """
    zeroes = (df == 0).sum()
    empty_strings = (df.replace(r"^\s*$", "", regex=True) == "").sum()
    nas = df.isna().sum()
    combined_counts = pd.DataFrame({
        "Zeroes": zeroes,
        "Empty Strings": empty_strings,
        "NaN": nas
        })
    return combined_counts

def find_outliers(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """
    Detects outliers in multiple features using the IQR method.

    Args:
        df: DataFrame containing the data.
        features: List of features to detect outliers in.

    Returns:
        DataFrame containing the outliers for each feature.
    """
    outliers_list = []
    for feature in features:
        if feature not in df.columns:
            print(f"Feature '{feature}' not found in DataFrame.")
            continue

        q1 = df[feature].quantile(0.25)
        q3 = df[feature].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        feature_outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
        if not feature_outliers.empty:
            print(f"Outliers in '{feature}':")
            print(feature_outliers[feature], end="\n\n")
            outliers_list.append(feature_outliers)
        else:
            print(f"No outliers in '{feature}'")

    if outliers_list:
        outliers = pd.concat(outliers_list)
        outliers = outliers[features]
    else:
        outliers = pd.DataFrame(columns=features)
        
    return outliers


def plot_sales_distribution(df: pd.DataFrame) -> None:
    """
    Plot the distribution of sales for each promotion using a box plot.
    
    Args:
        df: The DataFrame containing sales data with 'Promotion' and 'SalesInThousands' columns.
    
    Returns:
        None
    """
    sns.boxplot(df, hue="Promotion", y="SalesInThousands")
    plt.title("Sales Distribution by Promotion")
    plt.xlabel("Promotion")
    plt.ylabel("Sales in Thousands")
    plt.axhline(df['SalesInThousands'].median(), color='red',
                linestyle='--', label='Overall Median')
    plt.legend()
    plt.show()

def plot_promotion_histograms(df: pd.DataFrame) -> None:
    """
    Plot histograms of sales for each promotion with mean sales highlighted.
    
    Args:
        df: The DataFrame containing sales data with 'Promotion' and 'SalesInThousands' columns.
    
    Returns:
        None
    """
    promotions = df["Promotion"].unique()
    promotions = promotions.sort_values()
    fig, axes = plt.subplots(1, len(promotions), figsize=(15, 5),
                            sharey=True, sharex=True)

    for i, promotion in enumerate(promotions):
        curr_promo = df[df["Promotion"] == promotion]
        curr_mean = curr_promo["SalesInThousands"].mean()

        sns.histplot(curr_promo["SalesInThousands"], bins=30, kde=True,
                    ax=axes[i], color=f"C{i}")
        axes[i].axvline(curr_mean, color="#0571b0", linestyle="--",
                        label=f"Mean: {curr_mean:.2f}")
        axes[i].set_title(f"Promotion {promotion}")
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Frequency" if i == 0 else "")
        axes[i].legend()

    fig.supxlabel("Sales in Thousands")
    fig.suptitle("Sales Distribution by Promotion")
    plt.tight_layout()
    plt.show()

def calculate_summary(df: pd.DataFrame, group_by: List[str], targets: List[str],
                    to_percentage: bool = False) -> pd.DataFrame:
    """
    Calculates summary statistics and confidence intervals for grouped data.

    Args:
        df: The DataFrame containing the data.
        group_by: List of columns to group by.
        targets: List of target columns to calculate statistics for.
        to_percentage: Whether to express mean and confidence intervals as percentages.

    Returns:
        DataFrame with summary statistics and confidence intervals.
    """
    summaries = []
    for target in targets:
        summary = (df.groupby(group_by, observed=False)[target]
                .agg(["mean", "std", "count"]).reset_index())
        summary["metric"] = target
        if to_percentage:
            summary["mean"] *= 100
            summary["std"] *= 100
        summary["MoE"] = 1.96 * (summary["std"] / summary["count"]**0.5)
        summary["lower_ci"] = summary["mean"] - summary["MoE"]
        summary["upper_ci"] = summary["mean"] + summary["MoE"]
        summaries.append(summary)
    
    result = pd.concat(summaries, ignore_index=True)
    order = group_by + ["metric", "mean", "MoE", "lower_ci",
                        "upper_ci", "std", "count"]
    result = result[order]

    return result


def plot_sales_data(df: pd.DataFrame) -> None:
    """
    Plot mean sales by promotion with 95% confidence intervals.
    
    Args:
        df: The summary DataFrame containing 'Promotion', 'mean', and 'MoE' columns.
    
    Returns:
        None
    """
    sns.barplot(data=df, x="Promotion", y="mean", hue="Promotion",
            legend=False, palette=sns.color_palette(n_colors=3)
            )
    plt.errorbar(
        x=range(len(df)),
        y=df["mean"],
        yerr=df["MoE"],
        c="#0571b0",
        fmt="none",
        capsize=5,
    )
    plt.title("Mean Sales by Promotion with 95% Confidence Intervals")
    plt.ylabel("Mean Sales (in Thousands)")
    plt.xlabel("Promotion")
    plt.show()


def plot_market_sales_data(df: pd.DataFrame) -> None:
    """
    Plot mean sales by promotion for each market with 95% confidence intervals.
    
    Args:
        df: The summary DataFrame containing 'MarketID', 'Promotion', 'mean', and 'MoE' columns.
    
    Returns:
        None
    """
    fig, axes = plt.subplots(2, 5, figsize=(20,10), sharey=True)
    axes = axes.flatten()

    markets = df["MarketID"].unique()
    for i, market in enumerate(markets):
        ax = axes[i]
        data = df[df["MarketID"] == market]

        sns.barplot(ax=ax, data=data, x="Promotion", y="mean", hue="Promotion",
                    legend=False, palette=sns.color_palette(n_colors=3)
                    )
        
        ax.errorbar(
        x=range(len(data)),
        y=data["mean"],
        yerr=data["MoE"],
        c="#0571b0",
        fmt="none",
        capsize=5,
        )

        ax.set_title(f"Market {market}")
        ax.set_xlabel("")
        ax.set_ylabel("")

    fig.supxlabel("Promotion")
    fig.supylabel("Mean Sales (in Thousands)")
    fig.suptitle("Mean Sales by Market and Promotion with 95% Confidence Intervals")
    plt.tight_layout()
    plt.show()

def plot_gamerounds_boxplot(df: pd.DataFrame) -> None:
    """
    Plots a boxplot comparing the distribution of game rounds played across different versions.

    Args:
        df: The DataFrame containing the data to plot.

    Returns:
        None
    """
    sns.boxplot(df, x="sum_gamerounds", hue="version")
    plt.title("Game Rounds Played by Version")
    plt.ylabel("Game version")
    plt.xlabel("Game rounds played")
    plt.legend(title="Version",labels=["Gate 30", "Gate 40"])
    plt.show()

def plot_retention_rates(df: pd.DataFrame) -> None:
    """
    Plots a bar chart with error bars to display retention rates by version and metric 
    (e.g., 1-day and 7-day retention rates) with 95% confidence intervals.

    Args:
        df: The DataFrame containing the data to plot. 

    Returns:
        None
    """
    df["version"] = df["version"].cat.rename_categories({
    "gate_30": "Gate 30",
    "gate_40": "Gate 40"
    })
    barplot = sns.barplot(data=df, x="version", y="mean", hue="metric",
                          palette=sns.color_palette(n_colors=2)
                          )
    for bar, (_, row) in zip(barplot.patches, df.iterrows()):
        plt.errorbar(
            x=bar.get_x() + bar.get_width() / 2,
            y=row["mean"],
            yerr=row["MoE"],
            c="#0571b0",
            fmt="none",
            capsize=5,
        )
    plt.title("Retention Rates by Group with 95% Confidence Intervals")
    plt.xlabel("Group")
    plt.ylabel("Retention Rate (%)")
    plt.ylim(0, 100)
    handles, _ = plt.gca().get_legend_handles_labels()
    plt.legend(handles, ["1 day", "7 days"], title="Retention Period")
    plt.tight_layout()
    plt.show()

def get_variant_summary(df: pd.DataFrame, group_by: str) -> pd.DataFrame:
    """
    Summarizes sample sizes and proportions for each variant.

    Args:
        df: The DataFrame containing the data.
        group_by: The column to group variants by.

    Returns:
        A tuple containing:
            - Total sample size.
            - Variant counts as a Series.
            - Variant proportions as a DataFrame.
    """
    total_samples = len(df)
    print(f"Total Sample Size: {total_samples}\n")

    variant_counts = df[f"{group_by}"].value_counts()
    variant_proportions = variant_counts / total_samples * 100
    variant_summary = pd.DataFrame({
        "Sample Size": variant_counts,
        "Proportion": variant_proportions
    }).sort_index()

    return total_samples, variant_counts, variant_summary

def get_srm(total_samples: int, variant_counts: pd.Series) -> None:
    """
    Conducts a Sample Ratio Mismatch (SRM) test using the chi-square statistic.

    Args:
        total_samples: Total number of samples.
        variant_counts: Series of counts for each variant.

    Returns:
        None
    """
    alpha = 0.05

    expected = [total_samples / len(variant_counts)] * len(variant_counts)
    chi_score, p_value = chisquare(f_obs=variant_counts, f_exp=expected)

    print(f"Chi-Square statistic: {chi_score:.2f}")
    print(f"P-value: {p_value:.4f}")

    if p_value < alpha:
        print(f"The p-value is less than {alpha}, indicating a significant sample ratio mismatch.")
    else:
        print(f"The p-value is greater than {alpha}, indicating no significant sample ratio mismatch.")

def binomial_test(experiment: pd.Series, control: pd.Series) -> binomtest:
    """
    Performs a binomial test to compare experimental and control groups.

    Args:
        experiment: Series representing the experimental group outcomes.
        control: Series representing the control group outcomes.

    Returns:
        A binomtest result object.
    """
    combined_prob = (experiment.sum() + control.sum()) / (len(experiment) 
                                                          + len(control))
    return binomtest(
        k=experiment.sum(), 
        n=len(experiment), 
        p=combined_prob
    )

def prop_z_test(control: pd.Series, experiment: pd.Series) -> Tuple[float, float]:
    """
    Performs a two-proportion z-test for comparing control and experimental groups.

    Args:
        control: Series representing the control group outcomes.
        experiment: Series representing the experimental group outcomes.

    Returns:
        A tuple containing the z-statistic and p-value.
    """
    count = np.array([control.sum(), experiment.sum()])
    nobs = np.array([len(control), len(experiment)])
    return proportions_ztest(count, nobs, alternative='two-sided')

def bootstrap_ci(control: pd.Series, experiment: pd.Series) -> Tuple[float, float]:
    """
    Estimates confidence intervals for the mean difference using bootstrap sampling.

    Args:
        control: Series representing the control group outcomes.
        experiment: Series representing the experimental group outcomes.

    Returns:
        A tuple containing the lower and upper bounds of the confidence interval.
    """
    n_bootstraps = 10000
    differences = []

    for _ in range(n_bootstraps):
        control_sample = control.sample(len(control), replace=True)
        experiment_sample = experiment.sample(len(experiment), replace=True)
        diff = experiment_sample.mean() - control_sample.mean()
        differences.append(diff)

    ci_lower = np.percentile(differences, 2.5)
    ci_upper = np.percentile(differences, 97.5)
    return ci_lower, ci_upper

def plot_gamerounds_hist(df: pd.DataFrame) -> None:
    """
    Plots histograms of gamerounds distribution for different versions.

    Args:
        df: The DataFrame containing gamerounds and version columns.

    Returns:
        None
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    sns.histplot(df[df["version"] == "gate_30"], 
                x="sum_gamerounds", bins=100, kde=True, 
                ax=axes[0], color="C0")
    axes[0].set_title("Gate 30 gameround distribution")
    axes[0].set_xlabel("Gamerounds")
    axes[0].set_ylabel("Frequency")

    sns.histplot(df[df["version"] == "gate_40"], 
                x="sum_gamerounds", bins=100, kde=True, 
                ax=axes[1], color="C1")
    axes[1].set_title("Gate 40 gameround distribution")
    axes[1].set_xlabel("Gamerounds")

    fig.suptitle("Gameround distribution by version")
    plt.tight_layout()
    plt.show()