import pandas as pd
import numpy as np
import scipy
from scipy import stats

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import scikit_posthocs as sp
from sklearn import linear_model

def get_benchmark_aggregate(
    df, cols_groupby=["imputer", "predictor"], agg_func=pd.DataFrame.mean, keep_values=False
):
    metrics = [col for col in df.columns if "_score_" in col]
    durations = [col for col in df.columns if "duration_" in col]
    if cols_groupby is None:
        cols_groupby = [col for col in df.columns if col not in metrics and col not in durations]
    df_groupby = df.groupby(cols_groupby)[metrics + durations].apply(agg_func)

    if keep_values:
        for metric in metrics:
            df_groupby[f"{metric}_values"] = df.groupby(cols_groupby)[metric].apply(list)
        for duration in durations:
            df_groupby[f"{duration}_values"] = df.groupby(cols_groupby)[duration].apply(list)
    cols_imputation = [col for col in df_groupby.columns if "imputation_score_" in col]
    cols_prediction = [col for col in df_groupby.columns if "prediction_score_" in col]
    cols_train_set = [col for col in df_groupby.columns if "_train_set" in col]
    cols_test_set = [col for col in df_groupby.columns if "_test_set" in col]

    cols_duration_imputation = [col for col in df_groupby.columns if "_imputation_" in col]
    cols_duration_prediction = [col for col in df_groupby.columns if "_prediction_" in col]

    cols_multi_index = []
    for col in df_groupby.columns:
        if col in cols_imputation and col in cols_train_set:
            cols_multi_index.append(
                (
                    "imputation_score",
                    "train_set",
                    col.replace("imputation_score_", "").replace("_train_set", ""),
                )
            )
        if col in cols_imputation and col in cols_test_set:
            cols_multi_index.append(
                (
                    "imputation_score",
                    "test_set",
                    col.replace("imputation_score_", "").replace("_test_set", ""),
                )
            )
        if col in cols_prediction:
            if "notnan" in col:
                cols_multi_index.append(
                    (
                        "prediction_score",
                        "test_set_not_nan",
                        col.replace("prediction_score_notnan_", ""),
                    )
                )
            else:
                cols_multi_index.append(
                    (
                        "prediction_score",
                        "test_set_with_nan",
                        col.replace("prediction_score_nan_", ""),
                    )
                )
        if col in cols_duration_imputation:
            cols_multi_index.append(
                (
                    "duration",
                    "imputation",
                    col.replace("duration_imputation_", ""),
                )
            )
        if col in cols_duration_prediction:
            cols_multi_index.append(
                (
                    "duration",
                    "prediction",
                    col.replace("duration_prediction_", ""),
                )
            )

    df_groupby.columns = pd.MultiIndex.from_tuples(cols_multi_index)
    return df_groupby

def get_confidence_interval(x, confidence_level=0.95):
    # https://www.statology.org/confidence-intervals-python/
    interval = scipy.stats.norm.interval(
        confidence=confidence_level, loc=np.mean(x), scale=scipy.stats.sem(x)
    )
    width = interval[1] - interval[0]
    width_plus = interval[1] - np.mean(x)
    width_minus = np.mean(x) - interval[0]
    return [interval[0], interval[1], width, width_plus, width_minus]


def plot_bar_y_1D(
    df_agg,
    col_displayed=("prediction_score", "test_set", "wmape"),
    cols_grouped=["hole_generator", "imputer", "predictor"],
    add_annotation=True,
    add_confidence_interval=False,
    confidence_level=0.95,
    title="",
    title_x="",
    title_y="",
    title_legend="",
):
    df_agg_plot = df_agg.reset_index()
    col_legend = cols_grouped[-1]
    cols_x = [col for col in cols_grouped if col != col_legend]

    fig = go.Figure()
    for value in df_agg_plot[col_legend].unique():
        df_agg_plot_ = df_agg_plot[df_agg_plot[col_legend] == value]

        error_y = None
        if add_confidence_interval:
            value_ = list(col_displayed)
            value_[2] = value_[2] + "_values"
            error_y = np.array(
                df_agg_plot_.loc[:, tuple(value_)]
                .apply(lambda x: get_confidence_interval(x, confidence_level))
                .to_list()
            )

            # error_y_width = dict(type="data", array=error_y[:, 3] / 2)
            error_y_plus = error_y[:, 3]
            array_y_minus = error_y[:, 4]

            # error_y = error_y_width
            error_y = dict(
                type="data", symmetric=False, array=error_y_plus, arrayminus=array_y_minus
            )

        text = None
        if add_annotation:
            text = df_agg_plot_.loc[:, col_displayed]

        hovertemplate = f"{title_y}: "
        fig.add_trace(
            go.Bar(
                x=np.squeeze([df_agg_plot_[col].astype(str).values for col in cols_x]),
                y=df_agg_plot_.loc[:, col_displayed],
                showlegend=True,
                name=str(value),
                text=text,
                error_y=error_y,
                hovertemplate=hovertemplate + '%{y:.5f}<extra></extra>',
            )
        )
    metric_name = col_displayed[2]
    if add_annotation:
        fig.update_traces(texttemplate="%{text:.2}", textposition="outside")
    fig.update_layout(barmode="group")
    title_ = f'{metric_name} as a function of {"+".join(cols_grouped)}'
    if title != "":
        title_ = f"{title}, {title_}"
    fig.update_layout(title=title_)
    fig.update_xaxes(title="+".join(cols_grouped[:-1]))
    fig.update_layout(legend_title_text=str(cols_grouped[-1]))
    
    if title_x != "":
        fig.update_xaxes(title=title_x)
    if title_y != "":
        fig.update_yaxes(title=title_y)
    if title_legend != "":
        fig.update_layout(legend_title_text=title_legend)
    

    return fig


def plot_bar_y_nD(
    df_agg,
    cols_displayed=[
        ("imputation_score", "test_set", "wmape"),
        ("prediction_score", "test_set", "wmape"),
    ],
    cols_grouped=["hole_generator", "imputer", "predictor"],
    add_annotation=True,
    add_confidence_interval=False,
    confidence_level=0.95,
    title="",
    title_x="",
    title_y="",
    title_legend=""
):
    col_legend_idx = []
    for i in range(len(cols_displayed) - 1):
        for j in range(len(cols_displayed[i])):
            if cols_displayed[i][j] != cols_displayed[i + 1][j]:
                col_legend_idx.append(j)

    # fig = go.Figure()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for idx, value in enumerate(cols_displayed):
        name = "_".join([value[i] for i in set(col_legend_idx)])
        if "prediction" in name:
            secondary_y = False
        else:
            secondary_y = True
        offsetgroup = idx

        error_y = None
        if add_confidence_interval:
            value_ = list(value)
            value_[2] = value[2] + "_values"

            error_y = np.array(
                df_agg.loc[:, tuple(value_)]
                .apply(lambda x: get_confidence_interval(x, confidence_level))
                .to_list()
            )
            # error_y_width = dict(type="data", array=error_y[:, 2] / 2)
            error_y_plus = error_y[:, 3]
            array_y_minus = error_y[:, 4]

            # error_y = error_y_width
            error_y = dict(
                type="data", symmetric=False, array=error_y_plus, arrayminus=array_y_minus
            )

        text = None
        if add_annotation:
            text = df_agg.loc[:, value]

        fig.add_trace(
            go.Bar(
                name=name,
                x=np.array(df_agg.index.to_list()).transpose(),
                y=df_agg.loc[:, value],
                text=text,
                offsetgroup=offsetgroup,
                error_y=error_y,
            ),
            secondary_y=secondary_y,
        )

    metric_names = set([col[2] for col in cols_displayed])

    if add_annotation:
        fig.update_traces(texttemplate="%{text:.2}", textposition="outside")
    fig.update_layout(barmode="group")

    col_y_inter = set(cols_displayed[0])
    for s in cols_displayed[1:]:
        col_y_inter.intersection_update(s[:2])
    if len(col_y_inter) != 0:
        title_ = f'{" and ".join(metric_names)} as a function of {"+".join(cols_grouped)}'
        title_ += f'for {"+".join(list(col_y_inter))}'
    else:
        title_ = f'{" and ".join(metric_names)} as a function of {"+".join(cols_grouped)}'
    if title != "":
        title_ = f"{title}, {title_}"
    fig.update_layout(title=title_)
    type_names = "_".join(set([col[0] for col in cols_displayed]))
    if "prediction_score" in type_names:
        fig.update_yaxes(title_text="prediction_score", secondary_y=False)
    fig.update_yaxes(title_text="imputation_score", secondary_y=True)
    fig.update_xaxes(title="+".join(cols_grouped))
    fig.update_layout(legend_title_text="Options")
    
    if title_x != "":
        fig.update_xaxes(title=title_x)
    if title_y != "":
        fig.update_yaxes(title=title_y)
    if title_legend != "":
        fig.update_layout(legend_title_text=title_legend)

    return fig


def plot_bar(
    df,
    col_displayed=("prediction_score", "test_set", "wmape"),
    cols_displayed=None,
    cols_grouped=["hole_generator", "imputer", "predictor"],
    add_annotation=True,
    add_confidence_interval=False,
    confidence_level=0.95,
    title="",
    agg_func=pd.DataFrame.mean,
    title_x="",
    title_y="",
    title_legend="",
    yaxes_type="-",
):
    df_agg = get_benchmark_aggregate(
        df, cols_groupby=cols_grouped, agg_func=agg_func, keep_values=True
    )

    if cols_displayed is None:
        fig = plot_bar_y_1D(
            df_agg,
            col_displayed,
            cols_grouped,
            add_annotation,
            add_confidence_interval,
            confidence_level,
            title=title,
            title_x=title_x,
            title_y=title_y,
            title_legend=title_legend,
        )
    else:
        fig = plot_bar_y_nD(
            df_agg,
            cols_displayed,
            cols_grouped,
            add_annotation,
            add_confidence_interval,
            confidence_level,
            title=title,
            title_x=title_x,
            title_y=title_y,
            title_legend=title_legend
        )

    fig.update_yaxes(type=yaxes_type)
    # fig.update_layout(hovermode="x")

    return fig


def plot_scatter(
    df,
    cond={},
    col_x="imputation_score_mae_test_set",
    col_y="prediction_score_nan_mae",
    col_legend="ratio_masked",
    add_trend_line=True,
    model=linear_model.LinearRegression(),
):

    df_plot = df.copy()
    for k, v in cond.items():
        df_plot = df_plot[df_plot[k] == v]

    df_plot = df_plot.dropna()

    fig = go.Figure()
    for value in df_plot[col_legend].unique():
        df_plot_ = df_plot[df_plot[col_legend] == value]
        fig.add_trace(
            go.Scatter(
                x=df_plot_[col_x],
                y=df_plot_[col_y],
                name=str(value),
                mode="markers",
            )
        )

    if add_trend_line:
        model.fit(df_plot[[col_x]], df_plot[col_y])
        df_plot[f"{col_y}_predict"] = model.predict(df_plot[[col_x]])
        fig.add_trace(
            go.Scatter(
                x=df_plot[col_x],
                y=df_plot[f"{col_y}_predict"],
                name="trend",
                mode="lines",
                marker=dict(color="black"),
            )
        )

    fig.update_layout(legend_title=col_legend)
    fig.update_xaxes(title=col_x)
    fig.update_yaxes(title=col_y)
    title = f"{col_y} as a function of {col_x}"
    if len(cond) != 0:
        title += "<br>for "
        for k, v in cond.items():
            title += f"{k}={v}, "
    fig.update_layout(title=title[:-2])

    return fig


def get_relative_score(
    x, df, col, method="gain", ref_imputer="None", is_ref_hole_generator_none=False
):
    # https://en.wikipedia.org/wiki/Relative_change
    x_row = x[col]
    if is_ref_hole_generator_none:
        x_ref = df[
            (df["dataset"] == x["dataset"])
            & (df["n_fold"] == x["n_fold"])
            & (df["hole_generator"] == "None")
            & (df["predictor"] == x["predictor"])
            & (df["imputer"] == "None")
        ][col]
    else:
        if x["hole_generator"] == "None":
            x_ref = df[
                (df["dataset"] == x["dataset"])
                & (df["n_fold"] == x["n_fold"])
                & (df["hole_generator"] == "None")
                & (df["ratio_masked"] == x["ratio_masked"])
                & (df["predictor"] == x["predictor"])
                & (df["imputer"] == "None")
            ][col]
        else:
            x_ref = df[
                (df["dataset"] == x["dataset"])
                & (df["n_fold"] == x["n_fold"])
                & (df["hole_generator"] == x["hole_generator"])
                & (df["ratio_masked"] == x["ratio_masked"])
                & (df["n_mask"] == x["n_mask"])
                & (df["predictor"] == x["predictor"])
                & (df["imputer"] == ref_imputer)
            ][col]

    if method == "relative_percentage_gain":
        x_out = ((x_ref - x_row)) / x_ref
    elif method == "gain":
        x_out = x_ref - x_row
    else:
        x_out = x_row - x_ref
    return x_out.values


def statistic_test(
    df,
    col_evaluated,
    cols_grouped=[
        "dataset",
        "n_fold",
        "hole_generator",
        "ratio_masked",
        "n_mask",
        "predictor",
        "imputer",
    ],
    cols_displayed=["ratio_masked", "predictor"],
    func=stats.friedmanchisquare,
):
    df_values = df.groupby(cols_grouped)[col_evaluated].aggregate("first").unstack()
    cols_displayed_ = cols_displayed
    values = df_values.copy()

    def get_value(values, df_values, cols_displayed):
        col = cols_displayed[0]
        if len(cols_displayed) > 1:
            cols_displayed.remove(cols_displayed[0])
            list_df = []
            for v in df_values.index.get_level_values(col).unique():
                df_out = get_value(df_values.xs(v, level=col), df_values, cols_displayed)
                df_out[col] = v
                list_df.append(df_out)

            df_out = pd.concat(list_df)
            first_col = df_out.pop(col)
            df_out.insert(0, col, first_col)
            return df_out
        else:
            list_out = []
            for v in df_values.index.get_level_values(col).unique():
                values_ = values.xs(v, level=col).values.T
                res = func(*values_)
                list_out.append(
                    {
                        col: v,
                        "statistic": res.statistic,
                        "pvalue": res.pvalue,
                        "set_size": np.shape(values_),
                    }
                )
            df_out = pd.DataFrame(list_out)
            return df_out

    return get_value(values, df_values, cols_displayed_)


def plot_critical_difference_diagram(
    df, col_model, col_rank, col_value, title="", color_palette=None, fig_size=(7, 2)
):
    df_avg_rank = df.groupby(col_model)[col_rank].mean()
    df_values = df.groupby(col_model)[col_value].apply(list)
    model_names = df_avg_rank.index

    df_posthoc_conover_friedman = sp.posthoc_conover_friedman(np.array(list(df_values.values)).T)

    df_posthoc_conover_friedman.index = model_names
    df_posthoc_conover_friedman.columns = model_names

    if color_palette is None:
        color_palette = dict(
            [(key, value) for key, value in zip(model_names, np.random.rand(len(model_names), 3))]
        )
    figure = plt.figure(figsize=fig_size)
    plt.title(title)
    _ = sp.critical_difference_diagram(
        df_avg_rank, df_posthoc_conover_friedman, color_palette=color_palette
    )

    return figure
