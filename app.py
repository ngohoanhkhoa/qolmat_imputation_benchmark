import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import imputer_predictor as imppred

st.set_page_config(
    page_title="Imputation Benchmark for Tabular Data",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': 'https://github.com/scikit-learn-contrib/qolmat',
    }
)

st.title("Imputation Benchmark for Tabular Data")

results_plot = pd.read_pickle("benchmark_plot_app.pkl")

# st.dataframe(data)
num_dataset_full = len(results_plot["dataset"].unique())
num_predictor_full = len(results_plot["predictor"].unique())
num_imputer = len(results_plot["imputer"].unique()) - 1
num_fold = len(results_plot["n_fold"].unique())
# We remove the case [hole_generator=None, ratio_masked=0, n_mask=nan]
num_hole_generator = 1
num_mask = len(results_plot["n_mask"].unique()) - 1
num_ratio_masked = len(results_plot["ratio_masked"].unique()) - 1
num_trial = num_fold * num_mask

with st.sidebar:
    models = st.multiselect(
        "Model",
        ["HistGradientBoostingRegressor", "XGBRegressor", "Ridge"],
        ["HistGradientBoostingRegressor", "XGBRegressor", "Ridge"],
    )

    datasets = st.multiselect(
        "Dataset",
        [
            "sulfur",
            "wine_quality",
            "MiamiHousing2016",
            "elevators",
            "houses",
            "Brazilian_houses",
            "Bike_Sharing_Demand",
            "diamonds",
            "medical_charges",
        ],
        [
            "sulfur",
            "wine_quality",
            "MiamiHousing2016",
            "elevators",
            "houses",
            "Brazilian_houses",
            "Bike_Sharing_Demand",
            "diamonds",
            "medical_charges",
        ],
    )
    num_dataset = num_dataset_full
    if len(datasets) != num_dataset_full:
        num_dataset = len(datasets)
    num_predictor = num_predictor_full
    if len(models) != num_predictor_full:
        num_predictor = len(models)

    add_confidence_interval = st.toggle("Add confidence interval")
    confidence_level = 0.95
    if add_confidence_interval:
        confidence_level = st.slider("Confidence level", 0.0, 1.0, 0.95, step=0.05)

cols_plot_name = {
    "Dataset": "dataset",
    "Hole Type": "hole_generator",
    "NaN Ratio": "ratio_masked",
    "Imputer": "imputer",
    "Predictor": "predictor",
}

cols_plot_num = {
    "Dataset": num_dataset,
    "Hole Type": num_hole_generator,
    "NaN Ratio": num_ratio_masked,
    "Imputer": num_imputer,
    "Predictor": num_predictor,
}


def get_title_values(cols_grouped, cols_full=cols_plot_name.keys()):
    cols_grouped_out = [cols_plot_name[col] for col in cols_grouped]
    num_all = (
        np.prod(
            [
                cols_plot_num[col]
                for col in cols_plot_num.keys()
                if col not in cols_grouped
            ]
        )
        * num_trial
    )
    list_num = (
        " * ".join(
            [
                f"{cols_plot_num[col]} {col}"
                for col in cols_full
                if col not in cols_grouped
            ]
        )
        + f"* {num_trial} Trials"
    )
    title_x = cols_grouped_plot[1]
    if len(cols_grouped) > 1:
        title_x = " + ".join(cols_grouped_plot[:-1])

    return cols_grouped_out, num_all, list_num, title_x

# ------------------------------------------------
#
# ------------------------------------------------

st.header("A. Benchmark Configuration")
st.markdown(
    """
- **Imputers**: 7 imputation methods implemented in [Qolmat](https://github.com/scikit-learn-contrib/qolmat) (I = 7).
    - Conditional Imputation: RPCA, EM, MICE, KNN, Diffusion
    - Constant/simple Imputation: Median, Shuffle
- **Regression Predictor**: Ridge, HistGradientBoostingRegressor, XGBRegressor
- **Number of datasets**: 9 datsets without any missing values. M = 9 \n
- **Number of realizations** (K-folds): The combinations of imputers/prediction models are trained and evaluated K times (on the K different train and test sets). K = 5 \n
- **Generation of missing values** for stimulating imputation:
    - Types of missing values: MCAR. T = 1
    - Ratios of missing values: 10%, 30%, 50%, 70%. R = 4
    - N masks for each type and each ratio: N = 5
- Number of trials: E = M * K * N = 9 * 5 * 5 = 255 trials
- Number of configurations for each imputer: C = T * R * P = 1 * 4 * 3 = 12 configurations

| Dataset                | #Rows   | #Variables |
|------------------------|---------|------------|
| Wine Quality           | ~6K     | 11         |
| Sulfur                 | ~10K    | 6          |
| Brazilian Houses       | ~10.6K  | 8          |
| MiamiHousing2016       | ~13K    | 14         |
| Elevators              | ~16K    | 16         |
| Bike Sharing Demand    | ~17K    | 6          |
| Houses                 | ~20K    | 8          |
| Diamonds               | ~53K    | 6          |
| Medical Charges        | ~165K   | 5          |

"""

)

# ------------------------------------------------
#
# ------------------------------------------------

st.header("B. Prediction Performance")

option_type_set = st.selectbox(
    "Test set", ("Complete", "Imputed"), key="option_type_set_pp"
)
if option_type_set == "Complete":
    type_set = "test_set_not_nan"
else:
    type_set = "test_set_with_nan"

with st.expander("Weighted Mean Absolute Percentage Error (WMAPE)"):
    st.markdown(
        """
    - $$\\frac{\sum_{i=1}^N |G_i - F_i|}{\sum_{i=1}^N |G_i|}$$ where
        - $G_i$ is the ground-truth value at the sample i-th.
        - $F_i$ is the predicted value for the sample i-th.
        - N is the size of test set.
    - Smaller WMAPE, better prediction performance.
    """
    )
    st.write("---")
    metric = "wmape"
    cols_grouped_plot = st.multiselect(
        "Elements",
        ["Dataset", "Hole Type", "NaN Ratio", "Imputer", "Predictor"],
        ["Dataset", "NaN Ratio", "Imputer"],
        max_selections=3,
        key="cols_grouped_wmape",
    )
    cols_grouped, num_all, list_num, title_x = get_title_values(cols_grouped_plot)
    title = f"Average prediction performance over {num_all} ({list_num})."
    if type_set == "test_set_with_nan":
        title += " Evaluation based on WMAPE computed on imputed test sets."
    if type_set == "test_set_not_nan":
        title += " Evaluation based on WMAPE computed on complete test sets."
    if add_confidence_interval:
        title += "\n" + f"With a confidence level of {confidence_level}"
    st.markdown(title)

    fig = imppred.plot_bar(
        results_plot[
            ~(results_plot["imputer"].isin(["None"]))
            & (results_plot["predictor"].isin(models))
            & (results_plot["dataset"].isin(datasets))
        ],
        col_displayed=("prediction_score", type_set, metric),
        cols_grouped=cols_grouped,
        add_annotation=True,
        add_confidence_interval=add_confidence_interval,
        confidence_level=confidence_level,
        agg_func=pd.DataFrame.mean,
        title_x=title_x,
        title_y="WMAPE",
        title_legend=cols_grouped_plot[-1],
    )

    fig.update_layout(title="Prediction performance")
    st.plotly_chart(fig)

with st.expander("Imputer Rank based on prediction performance WMAPE"):
    st.markdown(
        f"""
    - The imputers are ranked for each dataset, ratio of nans, trial (one fold and one mask), and predictor.
    - The total round number is {num_dataset * num_ratio_masked * num_trial} ({num_dataset} Dataset * {num_ratio_masked} Ratio of Nans * {num_trial} Trials)
    """
    )
    st.write("---")
    metric = "wmape_imputer_rank"
    cols_grouped_plot = st.multiselect(
        "Elements",
        ["Dataset", "Hole Type", "NaN Ratio", "Imputer", "Predictor"],
        ["NaN Ratio", "Imputer", "Predictor"],
        max_selections=3,
        key="cols_grouped_rank",
    )
    cols_grouped, num_all, list_num, title_x = get_title_values(cols_grouped_plot)
    title = f"Average ranks of imputeurs for {num_all} rounds ({list_num})."
    if type_set == "test_set_with_nan":
        title += " Evaluation based on prediction performance (WMAPE) computed on imputed test sets."
    if type_set == "test_set_not_nan":
        title += " Evaluation based on prediction performance (WMAPE) computed on complete test sets."
    if add_confidence_interval:
        title += "\n" + f"With a confidence level of {confidence_level}"
    st.markdown(title)
    fig = imppred.plot_bar(
        results_plot[
            ~(results_plot["imputer"].isin(["None"]))
            & (results_plot["predictor"].isin(models))
            & (results_plot["dataset"].isin(datasets))
        ],
        col_displayed=("prediction_score", type_set, metric),
        cols_grouped=cols_grouped,
        add_annotation=True,
        add_confidence_interval=add_confidence_interval,
        confidence_level=confidence_level,
        agg_func=pd.DataFrame.mean,
        title_x=title_x,
        title_y="Average rank",
        title_legend=cols_grouped_plot[-1],
    )

    fig.update_layout(title="")
    st.plotly_chart(fig)

with st.expander("Critical difference diagram of average score ranks"):
    st.markdown(
        f"""
    -The imputers are ranked for each dataset, ratio of nans, trial (one fold and one mask), and predictor.
    - The total round number is {num_dataset * num_ratio_masked * num_trial} ({num_dataset} Dataset * {num_ratio_masked} Ratio of Nans * {num_trial} Trials)
    - Critical difference diagram of average score ranks (with Conover’s test): A crossbar is between two pairs that do not show a statistically significant difference between them.
    """
    )

    ratio_masked = st.select_slider(
        "Ratio of NaN",
        options=[
            0.1,
            0.3,
            0.5,
            0.7,
        ],
    )

    metric_cdg = "wmape"
    title = f"Average ranks for prediction performance, Ratio of NaN = {ratio_masked}."
    if type_set == "test_set_with_nan":
        type_set_cdg = "nan"
        title += f" Evaluation based on WMAPE computed on imputed test sets."
    if type_set == "test_set_not_nan":
        type_set_cdg = "notnan"
        title += f" Evaluation based on WMAPE computed on complete test sets."
    st.markdown(title)

    color_palette = dict(
        [
            (key, value)
            for key, value in zip(
                results_plot["imputer"].unique(),
                np.random.rand(len(results_plot["imputer"].unique()), 3),
            )
        ]
    )

    results_plot_ = results_plot[
        ~(results_plot["hole_generator"].isin(["None"]))
        & ~(results_plot["imputer"].isin(["None"]))
        & (results_plot["ratio_masked"].isin([ratio_masked]))
        & (results_plot["predictor"].isin(models))
        & (results_plot["dataset"].isin(datasets))
    ].copy()

    out = imppred.plot_critical_difference_diagram(
        results_plot_,
        col_model="imputer",
        col_rank=f"prediction_score_{type_set_cdg}_{metric_cdg}_imputer_rank",
        col_value=f"prediction_score_{type_set_cdg}_{metric_cdg}",
        title="",
        color_palette=color_palette,
    )
    st.plotly_chart(out)

# ------------------------------------------------
#
# ------------------------------------------------

st.header("C. Prediction Performance Gain")
st.subheader("C.1. With/without Imputation")
st.markdown(
    """
We compute the performance gain received through imputation-based prediction compared to the baselines trained on a complete/incomplete training set.
- Can Imputation Approach the Prediction Performance of a Complete Training Set?
- Does Imputation Improve the Performance of Predictive Models that Support Missing Values?
\n
**Competitors**: Predictive models trained on a **imputed** training set\n
**Baselines**: Predictive models trained on a **complete/incomplete** training set
"""
)
option_baseline = st.selectbox(
    "Baselines", ("Complete", "Incomplete"), label_visibility="collapsed"
)

if option_baseline == "Complete":
    metric_ratio = "wmape_gain_ratio_data_complete"
    metric_gain = "wmape_relative_percentage_gain_data_complete"
    metric_wsr = "wmape_gain"

else:
    metric_ratio = "wmape_gain_ratio"
    metric_gain = "wmape_relative_percentage_gain"
    metric_wsr = "wmape_gain"

option_type_set = st.selectbox(
    "Test set", ("Complete", "Imputed"), key="option_type_set_ppg"
)
if option_type_set == "Complete":
    type_set = "test_set_not_nan"
else:
    type_set = "test_set_with_nan"

with st.expander("Ratio of runs"):
    st.markdown(
        """
    Ratio of runs where the prediction is better, in terms of WMAPE, with imputation.
    - $$\operatorname{WMAPE}_P(C) > \operatorname{WMAPE}_P(B)$$ where
        - $\operatorname{WMAPE}_P$ is $\operatorname{WMAPE}$ score of prediction performance
        - $B$ and $C$ respectively represent the baselines and the competitors.
    """
    )
    st.markdown(
        f"""
    - The total number of trials/runs is E = {num_fold * num_mask * num_dataset} ({num_fold} folds x {num_mask} masks x {num_dataset} datasets).
    - Higher trial ratio, more obvious improvement
    """
    )
    st.write("---")
    cols_grouped_plot = st.multiselect(
        "Elements",
        ["Dataset", "Hole Type", "NaN Ratio", "Imputer", "Predictor"],
        ["Hole Type", "NaN Ratio", "Imputer"],
        max_selections=3,
        key="cols_grouped_ratio",
    )
    cols_grouped, num_all, list_num, title_x = get_title_values(cols_grouped_plot)
    title = f"Ratio of runs (over {num_all} runs = {list_num}) where a gain of prediction performance is found."
    if type_set == "test_set_with_nan":
        title += " Evaluation based on WMAPE computed on imputed test sets."
    if type_set == "test_set_not_nan":
        title += " Evaluation based on WMAPE computed on complete test sets."
    if "_data_complete" in metric_ratio:
        title += "\n Baseline: the predictor is trained on a complete training set."
    else:
        title += " Baseline: the predictor is trained on an incomplete train set."
    st.markdown(title)
    fig = imppred.plot_bar(
        results_plot[
            ~(results_plot["imputer"].isin(["None"]))
            & (results_plot["predictor"].isin(models))
            & (results_plot["dataset"].isin(datasets))
        ],
        col_displayed=("prediction_score", type_set, metric_ratio),
        cols_grouped=cols_grouped,
        add_annotation=True,
        add_confidence_interval=add_confidence_interval,
        confidence_level=confidence_level,
        agg_func=pd.DataFrame.sum,
        title_x=title_x,
        title_y="Ratio of runs",
        title_legend=cols_grouped_plot[-1],
    )

    fig.update_layout(title="")
    st.plotly_chart(fig)

with st.expander("Mean Relative Percentage Gain (MRPG)"):
    st.markdown(
        """
    - $$\\frac{\operatorname{WMAPE}_P(B) - \operatorname{WMAPE}_P(C)}{\operatorname{WMAPE}_P(B)}$$ where
        - $\operatorname{WMAPE}_P$ is $\operatorname{WMAPE}$ score of prediction performance
        - $B$ and $C$ respectively represent the baselines and the competitors.
    - Larger MRPG, more obvious the improvement.
    """
    )
    st.write("---")
    cols_grouped_plot = st.multiselect(
        "Elements",
        ["Dataset", "Hole Type", "NaN Ratio", "Imputer", "Predictor"],
        ["Dataset", "NaN Ratio", "Imputer"],
        max_selections=3,
        key="cols_grouped_mrpg",
    )
    cols_grouped, num_all, list_num, title_x = get_title_values(cols_grouped_plot)
    title = f"Mean relative percentage gain of prediction performance over {num_all} trials ({list_num})."
    if type_set == "test_set_with_nan":
        title += " Evaluation based on WMAPE computed on imputed test sets."
    if type_set == "test_set_not_nan":
        title += " Evaluation based on WMAPE computed on complete test sets."
    if "_data_complete" in metric_ratio:
        title += " Baseline: the predictor is trained on a complete training set."
    else:
        title += " Baseline: the predictor is trained on an incomplete train set."
    st.markdown(title)
    fig = imppred.plot_bar(
        results_plot[
            ~(results_plot["imputer"].isin(["None"]))
            & (results_plot["predictor"].isin(models))
            & (results_plot["dataset"].isin(datasets))
        ],
        col_displayed=("prediction_score", type_set, metric_gain),
        cols_grouped=cols_grouped,
        add_annotation=True,
        add_confidence_interval=add_confidence_interval,
        confidence_level=confidence_level,
        agg_func=pd.DataFrame.mean,
        title_x=title_x,
        title_y="MRPG",
        title_legend=cols_grouped_plot[-1],
    )

    fig.update_layout(title="")
    st.plotly_chart(fig)

with st.expander("The Wilcoxon signed-rank test on WMAPE gains (WMAPE-G)"):
    st.markdown(
        """
    - $$\operatorname{WMAPE-G} = \operatorname{WMAPE}_P(B) - \operatorname{WMAPE}_P(C)$$ where
        - $\operatorname{WMAPE}_P$ is $\operatorname{WMAPE}$ score of prediction performance
        - $B$ and $C$ respectively represent the baselines and the competitors.
    """
    )
    st.markdown(
        f"""
    - The Wilcoxon signed-rank test ([Perez-Lebel et al., 2022](https://arxiv.org/abs/2202.10580)) for gains grouped by nan ratios and predictors.
        - Alternative hypothesis (the median is greater than zero): imputation improves the performance of these predictive models
        - If a p-value < 5%, the null hypothesis (the median is negative) can be rejected in favor of the alternative, with a confidence level of 95%.
    - The total number of trials is E = {num_fold * num_mask * num_dataset} ({num_fold} folds x {num_mask} masks x {num_dataset} datasets).
    """
    )

    if type_set == "test_set_with_nan":
        type_set_wsr = "nan"
        title = "- Evaluation based on WMAPE computed on imputed test sets."
    if type_set == "test_set_not_nan":
        type_set_wsr = "notnan"
        title = "- Evaluation based on WMAPE computed on complete test sets."
    st.markdown(title)
    results_plot_ = results_plot[
        ~(results_plot["imputer"].isin(["None"]))
        & (results_plot["predictor"].isin(models))
        & (results_plot["dataset"].isin(datasets))
    ].copy()
    results_plot_ = results_plot_.rename(
        columns={
            "ratio_masked": "NaN ratio",
            "predictor": "Predictor",
            "imputer": "Imputer",
        }
    )
    groupby_cols = ["NaN ratio", "Predictor", "Imputer"]
    num_runs = (
        results_plot_.groupby(groupby_cols)
        .count()[f"prediction_score_{type_set_wsr}_{metric_wsr}"]
        .max()
    )
    wilcoxon_test = pd.DataFrame(
        results_plot_.groupby(groupby_cols)
        .apply(
            lambda x: stats.wilcoxon(
                x[f"prediction_score_{type_set_wsr}_{metric_wsr}"],
                alternative="greater",
            ).statistic
        )
        .rename("Wilcoxon-Test Statistic")
    )
    wilcoxon_test["Wilcoxon-Test P-value"] = pd.DataFrame(
        results_plot_.groupby(groupby_cols).apply(
            lambda x: stats.wilcoxon(
                x[f"prediction_score_{type_set_wsr}_{metric_wsr}"],
                alternative="greater",
            ).pvalue
        )
    )

    pvalue_threshold = st.slider("P-value threshold", 0.0, 1.0, 0.05, step=0.05)
    wilcoxon_test_plot = wilcoxon_test[
        wilcoxon_test["Wilcoxon-Test P-value"] < pvalue_threshold
    ]
    st.dataframe(wilcoxon_test_plot)

st.subheader("C.2. Conditional imputation vs Simple imputation")

st.markdown(
    """
We compute the prediction performance gain received through conditional imputation compared to simple imputation trained on a incomplete training set.

**Competitors**: Predictive models trained on a training set imputed by **Conditional Imputer**\n
**Baselines**: Predictive models trained on a training set imputed by **Simple Imputer**
"""
)
ref_imputer = st.selectbox(
    "Baseline", ("ImputerMedian", "ImputerShuffle"), label_visibility="collapsed"
)

option_type_set = st.selectbox(
    "Test set", ("Complete", "Imputed"), key="option_type_set_cisi"
)
if option_type_set == "Complete":
    type_set = "test_set_not_nan"
else:
    type_set = "test_set_with_nan"

metric_gain = f"wmape_gain_{ref_imputer}"
metric_wsr = f"wmape_gain_{ref_imputer}"

if len(models) == 1:
    metric_ratio = f"wmape_gain_ratio_{ref_imputer}_each"
else:
    metric_ratio = f"wmape_gain_ratio_{ref_imputer}_all"

with st.expander("Ratio of runs"):
    st.markdown(
        """
    Ratio of runs where the prediction is better, in terms of WMAPE, with imputation.
    - $$\operatorname{WMAPE}_P(C) > \operatorname{WMAPE}_P(B)$$ where
        - $\operatorname{WMAPE}_P$ is $\operatorname{WMAPE}$ score of prediction performance
        - $B$ and $C$ respectively represent the baselines and the competitors.
    """
    )
    st.markdown(
        f"""
    - The total number of trials/runs is E = {num_fold * num_mask * num_dataset} ({num_fold} folds x {num_mask} masks x {num_dataset} datasets).
    - Higher trial ratio, more obvious improvement
    """
    )
    st.write("---")
    cols_grouped_plot = st.multiselect(
        "Elements",
        ["Dataset", "Hole Type", "NaN Ratio", "Imputer", "Predictor"],
        ["Hole Type", "NaN Ratio", "Imputer"],
        max_selections=3,
        key="cols_grouped_ratio_cisi",
    )
    cols_grouped, num_all, list_num, title_x = get_title_values(cols_grouped_plot)
    title = f"Ratio of runs (over {num_all} runs = {list_num}) where a gain of prediction performance is found."
    if type_set == "test_set_with_nan":
        title += " Evaluation based on WMAPE computed on imputed test sets."
    if type_set == "test_set_not_nan":
        title += " Evaluation based on WMAPE computed on complete test sets."
    st.markdown(title)
    fig = imppred.plot_bar(
        results_plot[
            ~(results_plot["imputer"].isin(["None", ref_imputer]))
            & (results_plot["predictor"].isin(models))
            & (results_plot["dataset"].isin(datasets))
        ],
        col_displayed=("prediction_score", type_set, metric_ratio),
        cols_grouped=cols_grouped,
        add_annotation=True,
        add_confidence_interval=add_confidence_interval,
        confidence_level=confidence_level,
        agg_func=pd.DataFrame.sum,
        title_x=title_x,
        title_y="Ratio of runs",
        title_legend=cols_grouped_plot[-1],
    )

    fig.update_layout(title="")
    st.plotly_chart(fig)

with st.expander("Mean Relative Percentage Gain (MRPG)"):
    st.markdown(
        """
    - $$\\frac{\operatorname{WMAPE}_P(B) - \operatorname{WMAPE}_P(C)}{\operatorname{WMAPE}_P(B)}$$ where
        - $\operatorname{WMAPE}_P$ is $\operatorname{WMAPE}$ score of prediction performance
        - $B$ and $C$ respectively represent the baselines and the competitors.
    - Larger MRPG, more obvious the improvement.
    """
    )
    st.write("---")
    cols_grouped_plot = st.multiselect(
        "Elements",
        ["Dataset", "Hole Type", "NaN Ratio", "Imputer", "Predictor"],
        ["Dataset", "NaN Ratio", "Imputer"],
        max_selections=3,
        key="cols_grouped_mrpg_cisi",
    )
    cols_grouped, num_all, list_num, title_x = get_title_values(cols_grouped_plot)
    title = f"Mean relative percentage gain of prediction performance over {num_all} trials ({list_num})."
    if type_set == "test_set_with_nan":
        title += " Evaluation based on WMAPE computed on imputed test sets."
    if type_set == "test_set_not_nan":
        title += " Evaluation based on WMAPE computed on complete test sets."
    if "_data_complete" in metric_ratio:
        title += " Baseline: the predictor is trained on a complete training set."
    else:
        title += " Baseline: the predictor is trained on an incomplete train set."
    st.markdown(title)
    fig = imppred.plot_bar(
        results_plot[
            ~(results_plot["imputer"].isin(["None"]))
            & (results_plot["predictor"].isin(models))
            & (results_plot["dataset"].isin(datasets))
        ],
        col_displayed=("prediction_score", type_set, metric_gain),
        cols_grouped=cols_grouped,
        add_annotation=True,
        add_confidence_interval=add_confidence_interval,
        confidence_level=confidence_level,
        agg_func=pd.DataFrame.mean,
        title_x=title_x,
        title_y="MRPG",
        title_legend=cols_grouped_plot[-1],
    )

    fig.update_layout(title="")
    st.plotly_chart(fig)

with st.expander("The Wilcoxon signed-rank test on WMAPE gains (WMAPE-G)"):
    st.markdown(
        """
    - $$\operatorname{WMAPE-G} = \operatorname{WMAPE}_P(B) - \operatorname{WMAPE}_P(C)$$ where
        - $\operatorname{WMAPE}_P$ is $\operatorname{WMAPE}$ score of prediction performance
        - $B$ and $C$ respectively represent the baselines and the competitors.
    """
    )
    st.markdown(
        f"""
    - The Wilcoxon signed-rank test ([Perez-Lebel et al., 2022](https://arxiv.org/abs/2202.10580)) for gains grouped by nan ratios and predictors.
        - Alternative hypothesis (the median is greater than zero): imputation improves the performance of these predictive models
        - If a p-value < 5%, the null hypothesis (the median is negative) can be rejected in favor of the alternative, with a confidence level of 95%.
    - The total number of trials is E = {num_fold * num_mask * num_dataset} ({num_fold} folds x {num_mask} masks x {num_dataset} datasets).
    """
    )

    if type_set == "test_set_with_nan":
        type_set_wsr = "nan"
        title = "- Evaluation based on WMAPE computed on imputed test sets."
    if type_set == "test_set_not_nan":
        type_set_wsr = "notnan"
        title = "- Evaluation based on WMAPE computed on complete test sets."
    st.markdown(title)
    results_plot_ = results_plot[
        ~(results_plot["imputer"].isin(["None", ref_imputer]))
        & (results_plot["predictor"].isin(models))
        & (results_plot["dataset"].isin(datasets))
    ].copy()
    results_plot_ = results_plot_.rename(
        columns={
            "ratio_masked": "NaN ratio",
            "predictor": "Predictor",
            "imputer": "Imputer",
        }
    )
    groupby_cols = ["NaN ratio", "Predictor", "Imputer"]
    num_runs = (
        results_plot_.groupby(groupby_cols)
        .count()[f"prediction_score_{type_set_wsr}_{metric_wsr}"]
        .max()
    )
    wilcoxon_test = pd.DataFrame(
        results_plot_.groupby(groupby_cols)
        .apply(
            lambda x: stats.wilcoxon(
                x[f"prediction_score_{type_set_wsr}_{metric_wsr}"],
                alternative="greater",
            ).statistic
        )
        .rename("Wilcoxon-Test Statistic")
    )
    wilcoxon_test["Wilcoxon-Test P-value"] = pd.DataFrame(
        results_plot_.groupby(groupby_cols).apply(
            lambda x: stats.wilcoxon(
                x[f"prediction_score_{type_set_wsr}_{metric_wsr}"],
                alternative="greater",
            ).pvalue
        )
    )

    pvalue_threshold = st.slider(
        "P-value threshold", 0.0, 1.0, 0.05, step=0.05, key="pvalue_threshold_cisi"
    )
    wilcoxon_test_plot = wilcoxon_test[
        wilcoxon_test["Wilcoxon-Test P-value"] < pvalue_threshold
    ]
    st.dataframe(wilcoxon_test_plot)

# ------------------------------------------------
#
# ------------------------------------------------

st.header("D. Imputation Performance")
st.markdown(
    """
"""
)
metric_plot = st.selectbox(
    "Imputation Performance Metric",
    ("WMAPE", "Distance Correlation Pattern"),
)
if metric_plot == "WMAPE":
    metric = "wmape"
else:
    metric = "dist_corr_pattern"

with st.expander("Mean Performance Score"):
    st.markdown(
        """
    """
    )
    st.write("---")

    cols_grouped_plot = st.multiselect(
        "Elements",
        ["Dataset", "Hole Type", "NaN Ratio", "Imputer"],
        ["Dataset", "NaN Ratio", "Imputer"],
        max_selections=3,
        key="cols_grouped_mps",
    )
    cols_grouped, num_all, list_num, title_x = get_title_values(
        cols_grouped_plot, cols_full=["Dataset", "Hole Type", "NaN Ratio", "Imputer"]
    )

    title = f"Mean imputation performance over {num_all} trials ({list_num})."

    type_set_plot = st.selectbox(
        "See the performance of training/test set",
        ("Training set", "Test set"),
    )
    if type_set_plot == "Training set":
        type_set = "train_set"
        title += (
            f" Evaluation based on {metric_plot} computed on imputed training sets."
        )
    else:
        type_set = "test_set"
        title += f" Evaluation based on {metric_plot} computed on imputed test sets."

    st.markdown(title)
    fig = imppred.plot_bar(
        results_plot[
            ~(results_plot["imputer"].isin(["None"]))
            & (results_plot["predictor"].isin(models))
            & (results_plot["dataset"].isin(datasets))
        ],
        col_displayed=("imputation_score", type_set, metric),
        cols_grouped=cols_grouped,
        add_annotation=True,
        add_confidence_interval=add_confidence_interval,
        confidence_level=confidence_level,
        agg_func=pd.DataFrame.mean,
        title_x=title_x,
        title_y=metric_plot,
        title_legend=cols_grouped_plot[-1],
    )

    fig.update_layout(title="")
    st.plotly_chart(fig)

with st.expander("Imputer Rank based on imputation performance"):
    st.markdown(
        f"""
    - The imputers are ranked for each dataset, each ratio of NaNs, each trial (one fold and one mask).
    - The total round number is {num_dataset * num_ratio_masked * num_trial} ({num_dataset} Dataset * {num_ratio_masked} Ratio of Nans * {num_trial} Trials)
    """
    )
    st.write("---")
    type_set_plot = st.selectbox(
        "See the performance of training/test set",
        ("Training set", "Test set", "Both"),
        key="type_set_plot_rank",
    )
    if type_set_plot == "Both":
        cols_grouped_plot = st.multiselect(
            "Elements",
            ["Dataset", "Hole Type", "NaN Ratio", "Imputer"],
            ["NaN Ratio", "Imputer"],
            max_selections=2,
            key="cols_grouped_rank_imputation",
        )
    else:
        cols_grouped_plot = st.multiselect(
            "Elements",
            ["Dataset", "Hole Type", "NaN Ratio", "Imputer"],
            ["NaN Ratio", "Imputer", "Dataset"],
            max_selections=3,
            key="cols_grouped_rank_imputation",
        )

    cols_grouped, num_all, list_num, title_x = get_title_values(
        cols_grouped_plot, cols_full=["Dataset", "Hole Type", "NaN Ratio", "Imputer"]
    )

    title = f"Average ranks of imputeurs for {num_all} rounds ({list_num})."
    if type_set_plot == "Training set":
        type_set = "train_set"
        title += f" Evaluation based on imputation performanc {metric_plot} computed on imputed training sets."
    elif type_set_plot == "Test set":
        type_set = "test_set"
        title += f" Evaluation based on imputation performance{metric_plot} computed on imputed test sets."
    else:
        type_set = ""
        title += f" Evaluation based on imputation performance {metric_plot} computed on imputed training/test sets."

    if add_confidence_interval:
        title += "\n" + f"With a confidence level of {confidence_level}"
    st.markdown(title)

    if type_set_plot == "Both":
        fig = imppred.plot_bar(
            results_plot[
                ~(results_plot["imputer"].isin(["None"]))
                & (results_plot["predictor"].isin(models))
                & (results_plot["dataset"].isin(datasets))
            ],
            cols_displayed=(
                ("imputation_score", "test_set", f"{metric}_rank"),
                ("imputation_score", "train_set", f"{metric}_rank"),
            ),
            cols_grouped=cols_grouped,
            add_annotation=True,
            add_confidence_interval=add_confidence_interval,
            confidence_level=confidence_level,
            agg_func=pd.DataFrame.mean,
            title_x=title_x,
            title_y="Average rank",
            title_legend=cols_grouped_plot[-1],
        )
    else:
        fig = imppred.plot_bar(
            results_plot[
                ~(results_plot["imputer"].isin(["None"]))
                & (results_plot["predictor"].isin(models))
                & (results_plot["dataset"].isin(datasets))
            ],
            col_displayed=("imputation_score", type_set, f"{metric}_rank"),
            cols_grouped=cols_grouped,
            add_annotation=True,
            add_confidence_interval=add_confidence_interval,
            confidence_level=confidence_level,
            agg_func=pd.DataFrame.mean,
            title_x=title_x,
            title_y="Average rank",
            title_legend=cols_grouped_plot[-1],
        )

    fig.update_layout(title="")
    st.plotly_chart(fig)

with st.expander("Critical difference diagram of average score ranks"):
    st.markdown(
        f"""
    - The imputers are ranked for each dataset, ratio of nans, trial (one fold and one mask).
    - The total round number is {num_dataset * num_ratio_masked * num_trial} ({num_dataset} Dataset * {num_ratio_masked} Ratio of Nans * {num_trial} Trials)
    - Critical difference diagram of average score ranks (with Conover’s test): A crossbar is between two pairs that do not show a statistically significant difference between them.
    """
    )

    ratio_masked = st.select_slider(
        "Ratio of NaN",
        options=[
            0.1,
            0.3,
            0.5,
            0.7,
        ],
        key="ratio_masked_imputation",
    )

    title = f"Average ranks for imputation performance ({metric}), Ratio of NaN = {ratio_masked}."
    type_set_plot = st.selectbox(
        "See the performance of training/test set",
        ("Training set", "Test set"),
        key="type_set_plot_imputation",
    )
    if type_set_plot == "Training set":
        type_set = "train_set"
        title += (
            f" Evaluation based on {metric_plot} computed on imputed training sets."
        )
    else:
        type_set = "test_set"
        title += f" Evaluation based on {metric_plot} computed on imputed test sets."

    st.markdown(title)

    color_palette = dict(
        [
            (key, value)
            for key, value in zip(
                results_plot["imputer"].unique(),
                np.random.rand(len(results_plot["imputer"].unique()), 3),
            )
        ]
    )

    results_plot_ = results_plot[
        ~(results_plot["hole_generator"].isin(["None"]))
        & ~(results_plot["imputer"].isin(["None"]))
        & (results_plot["ratio_masked"].isin([ratio_masked]))
        & (results_plot["predictor"].isin(models))
        & (results_plot["dataset"].isin(datasets))
    ].copy()

    out = imppred.plot_critical_difference_diagram(
        results_plot_,
        col_model="imputer",
        col_rank=f"imputation_score_{metric}_rank_{type_set}",
        col_value=f"imputation_score_{metric}_{type_set}",
        title="",
        color_palette=color_palette,
    )
    st.plotly_chart(out)

# ------------------------------------------------
#
# ------------------------------------------------

st.header("E. Prediction Performance of Pairs Imputer-Predictor")
st.markdown(
    """

"""
)
option_type_set = st.selectbox(
    "Test set", ("Complete", "Imputed"), key="option_type_set_pip"
)
if option_type_set == "Complete":
    type_set = "test_set_not_nan"
else:
    type_set = "test_set_with_nan"

with st.expander("Weighted Mean Absolute Percentage Error (WMAPE)"):
    st.markdown(
        """
    - $$\\frac{\sum_{i=1}^N |G_i - F_i|}{\sum_{i=1}^N |G_i|}$$ where
        - $G_i$ is the ground-truth value at the sample i-th.
        - $F_i$ is the predicted value for the sample i-th.
        - N is the size of test set.
    - Smaller WMAPE, better prediction performance.
    """
    )
    st.write("---")
    metric = "wmape"
    cols_grouped_plot = st.multiselect(
        "Elements",
        ["Dataset", "Hole Type", "NaN Ratio"],
        ["Dataset", "NaN Ratio"],
        max_selections=2,
        key="cols_grouped_wmape_pip",
    )
    cols_grouped, num_all, list_num, title_x = get_title_values(cols_grouped_plot, cols_full=["Dataset", "Hole Type", "NaN Ratio"])
    cols_grouped += ['imputer_predictor']
    title = f"Average prediction performance over {num_all} ({list_num})."
    if type_set == "test_set_with_nan":
        title += " Evaluation based on WMAPE computed on imputed test sets."
    if type_set == "test_set_not_nan":
        title += " Evaluation based on WMAPE computed on complete test sets."
    if add_confidence_interval:
        title += "\n" + f"With a confidence level of {confidence_level}"
    st.markdown(title)

    fig = imppred.plot_bar(
        results_plot[
            ~(results_plot["imputer"].isin(["None"]))
            & (results_plot["predictor"].isin(models))
            & (results_plot["dataset"].isin(datasets))
        ],
        col_displayed=("prediction_score", type_set, metric),
        cols_grouped=cols_grouped,
        add_annotation=True,
        add_confidence_interval=add_confidence_interval,
        confidence_level=confidence_level,
        agg_func=pd.DataFrame.mean,
        title_x=title_x,
        title_y="WMAPE",
        title_legend="Imp-Pre",
    )

    fig.update_layout(title="Prediction performance")
    st.plotly_chart(fig)

with st.expander("Imputer-Predictor Rank based on prediction performance WMAPE"):
    st.markdown(
        f"""
    - The pairs imputer-predictor are ranked for each dataset, ratio of nans, trial (one fold and one mask).
    - The total round number is {num_dataset * num_ratio_masked * num_trial} ({num_dataset} Dataset * {num_ratio_masked} Ratio of Nans * {num_trial} Trials)
    """
    )
    st.write("---")
    metric = "wmape_imputer_predictor_rank"
    cols_grouped_plot = st.multiselect(
        "Elements",
        ["Dataset", "Hole Type", "NaN Ratio", "Imputer", "Predictor"],
        ["Predictor", "NaN Ratio", "Imputer"],
        max_selections=3,
        key="cols_grouped_rank_pip",
    )
    cols_grouped, num_all, list_num, title_x = get_title_values(cols_grouped_plot)
    title = f"Average ranks of imputeurs for {num_all} rounds ({list_num})."
    if type_set == "test_set_with_nan":
        title += " Evaluation based on prediction performance (WMAPE) computed on imputed test sets."
    if type_set == "test_set_not_nan":
        title += " Evaluation based on prediction performance (WMAPE) computed on complete test sets."
    if add_confidence_interval:
        title += "\n" + f"With a confidence level of {confidence_level}"
    st.markdown(title)
    fig = imppred.plot_bar(
        results_plot[
            ~(results_plot['hole_generator'].isin(['None']))
            & (results_plot["predictor"].isin(models))
            & (results_plot["dataset"].isin(datasets))
        ],
        col_displayed=("prediction_score", type_set, metric),
        cols_grouped=cols_grouped,
        add_annotation=True,
        add_confidence_interval=add_confidence_interval,
        confidence_level=confidence_level,
        agg_func=pd.DataFrame.mean,
        title_x=title_x,
        title_y="Average rank",
        title_legend=cols_grouped_plot[-1],
    )

    fig.update_layout(title="")
    st.plotly_chart(fig)

with st.expander("Critical difference diagram of average score ranks"):
    st.markdown(
        f"""
    - The pair imputer-predictor are ranked for each dataset, ratio of nans, trial (one fold and one mask).
    - The total round number is {num_dataset * num_ratio_masked * num_trial} ({num_dataset} Dataset * {num_ratio_masked} Ratio of Nans * {num_trial} Trials)
    - Critical difference diagram of average score ranks (with Conover’s test): A crossbar is between two pairs that do not show a statistically significant difference between them.
    """
    )

    ratio_masked = st.select_slider(
        "Ratio of NaN",
        options=[
            0.1,
            0.3,
            0.5,
            0.7,
        ],
        key="ratio_masked_pip"
    )

    metric_cdg = "wmape"
    title = f"Average ranks for prediction performance, Ratio of NaN = {ratio_masked}."
    if type_set == "test_set_with_nan":
        type_set_cdg = "nan"
        title += f" Evaluation based on WMAPE computed on imputed test sets."
    if type_set == "test_set_not_nan":
        type_set_cdg = "notnan"
        title += f" Evaluation based on WMAPE computed on complete test sets."
    st.markdown(title)

    color_palette = dict(
        [
            (key, value)
            for key, value in zip(
                results_plot["imputer_predictor"].unique(),
                np.random.rand(len(results_plot["imputer_predictor"].unique()), 3),
            )
        ]
    )

    results_plot_ = results_plot[
        ~(results_plot["hole_generator"].isin(["None"]))
        & ~(results_plot["imputer"].isin(["None"]))
        & (results_plot["ratio_masked"].isin([ratio_masked]))
        & (results_plot["predictor"].isin(models))
        & (results_plot["dataset"].isin(datasets))
    ].copy()

    out = imppred.plot_critical_difference_diagram(
        results_plot_,
        col_model="imputer_predictor",
        col_rank=f"prediction_score_{type_set_cdg}_{metric_cdg}_imputer_predictor_rank",
        col_value=f"prediction_score_{type_set_cdg}_{metric_cdg}",
        title="",
        color_palette=color_palette,
    )
    st.plotly_chart(out)