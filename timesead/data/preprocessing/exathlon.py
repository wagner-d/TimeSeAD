# Most of the preprocessing code is taken from https://github.com/exathlonbenchmark/exathlon
#
# License:
#
#                                  Apache License
#                            Version 2.0, January 2004
#                         http://www.apache.org/licenses/
#
#    TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION
#
#    1. Definitions.
#
#       "License" shall mean the terms and conditions for use, reproduction,
#       and distribution as defined by Sections 1 through 9 of this document.
#
#       "Licensor" shall mean the copyright owner or entity authorized by
#       the copyright owner that is granting the License.
#
#       "Legal Entity" shall mean the union of the acting entity and all
#       other entities that control, are controlled by, or are under common
#       control with that entity. For the purposes of this definition,
#       "control" means (i) the power, direct or indirect, to cause the
#       direction or management of such entity, whether by contract or
#       otherwise, or (ii) ownership of fifty percent (50%) or more of the
#       outstanding shares, or (iii) beneficial ownership of such entity.
#
#       "You" (or "Your") shall mean an individual or Legal Entity
#       exercising permissions granted by this License.
#
#       "Source" form shall mean the preferred form for making modifications,
#       including but not limited to software source code, documentation
#       source, and configuration files.
#
#       "Object" form shall mean any form resulting from mechanical
#       transformation or translation of a Source form, including but
#       not limited to compiled object code, generated documentation,
#       and conversions to other media types.
#
#       "Work" shall mean the work of authorship, whether in Source or
#       Object form, made available under the License, as indicated by a
#       copyright notice that is included in or attached to the work
#       (an example is provided in the Appendix below).
#
#       "Derivative Works" shall mean any work, whether in Source or Object
#       form, that is based on (or derived from) the Work and for which the
#       editorial revisions, annotations, elaborations, or other modifications
#       represent, as a whole, an original work of authorship. For the purposes
#       of this License, Derivative Works shall not include works that remain
#       separable from, or merely link (or bind by name) to the interfaces of,
#       the Work and Derivative Works thereof.
#
#       "Contribution" shall mean any work of authorship, including
#       the original version of the Work and any modifications or additions
#       to that Work or Derivative Works thereof, that is intentionally
#       submitted to Licensor for inclusion in the Work by the copyright owner
#       or by an individual or Legal Entity authorized to submit on behalf of
#       the copyright owner. For the purposes of this definition, "submitted"
#       means any form of electronic, verbal, or written communication sent
#       to the Licensor or its representatives, including but not limited to
#       communication on electronic mailing lists, source code control systems,
#       and issue tracking systems that are managed by, or on behalf of, the
#       Licensor for the purpose of discussing and improving the Work, but
#       excluding communication that is conspicuously marked or otherwise
#       designated in writing by the copyright owner as "Not a Contribution."
#
#       "Contributor" shall mean Licensor and any individual or Legal Entity
#       on behalf of whom a Contribution has been received by Licensor and
#       subsequently incorporated within the Work.
#
#    2. Grant of Copyright License. Subject to the terms and conditions of
#       this License, each Contributor hereby grants to You a perpetual,
#       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#       copyright license to reproduce, prepare Derivative Works of,
#       publicly display, publicly perform, sublicense, and distribute the
#       Work and such Derivative Works in Source or Object form.
#
#    3. Grant of Patent License. Subject to the terms and conditions of
#       this License, each Contributor hereby grants to You a perpetual,
#       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#       (except as stated in this section) patent license to make, have made,
#       use, offer to sell, sell, import, and otherwise transfer the Work,
#       where such license applies only to those patent claims licensable
#       by such Contributor that are necessarily infringed by their
#       Contribution(s) alone or by combination of their Contribution(s)
#       with the Work to which such Contribution(s) was submitted. If You
#       institute patent litigation against any entity (including a
#       cross-claim or counterclaim in a lawsuit) alleging that the Work
#       or a Contribution incorporated within the Work constitutes direct
#       or contributory patent infringement, then any patent licenses
#       granted to You under this License for that Work shall terminate
#       as of the date such litigation is filed.
#
#    4. Redistribution. You may reproduce and distribute copies of the
#       Work or Derivative Works thereof in any medium, with or without
#       modifications, and in Source or Object form, provided that You
#       meet the following conditions:
#
#       (a) You must give any other recipients of the Work or
#           Derivative Works a copy of this License; and
#
#       (b) You must cause any modified files to carry prominent notices
#           stating that You changed the files; and
#
#       (c) You must retain, in the Source form of any Derivative Works
#           that You distribute, all copyright, patent, trademark, and
#           attribution notices from the Source form of the Work,
#           excluding those notices that do not pertain to any part of
#           the Derivative Works; and
#
#       (d) If the Work includes a "NOTICE" text file as part of its
#           distribution, then any Derivative Works that You distribute must
#           include a readable copy of the attribution notices contained
#           within such NOTICE file, excluding those notices that do not
#           pertain to any part of the Derivative Works, in at least one
#           of the following places: within a NOTICE text file distributed
#           as part of the Derivative Works; within the Source form or
#           documentation, if provided along with the Derivative Works; or,
#           within a display generated by the Derivative Works, if and
#           wherever such third-party notices normally appear. The contents
#           of the NOTICE file are for informational purposes only and
#           do not modify the License. You may add Your own attribution
#           notices within Derivative Works that You distribute, alongside
#           or as an addendum to the NOTICE text from the Work, provided
#           that such additional attribution notices cannot be construed
#           as modifying the License.
#
#       You may add Your own copyright statement to Your modifications and
#       may provide additional or different license terms and conditions
#       for use, reproduction, or distribution of Your modifications, or
#       for any such Derivative Works as a whole, provided Your use,
#       reproduction, and distribution of the Work otherwise complies with
#       the conditions stated in this License.
#
#    5. Submission of Contributions. Unless You explicitly state otherwise,
#       any Contribution intentionally submitted for inclusion in the Work
#       by You to the Licensor shall be under the terms and conditions of
#       this License, without any additional terms or conditions.
#       Notwithstanding the above, nothing herein shall supersede or modify
#       the terms of any separate license agreement you may have executed
#       with Licensor regarding such Contributions.
#
#    6. Trademarks. This License does not grant permission to use the trade
#       names, trademarks, service marks, or product names of the Licensor,
#       except as required for reasonable and customary use in describing the
#       origin of the Work and reproducing the content of the NOTICE file.
#
#    7. Disclaimer of Warranty. Unless required by applicable law or
#       agreed to in writing, Licensor provides the Work (and each
#       Contributor provides its Contributions) on an "AS IS" BASIS,
#       WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
#       implied, including, without limitation, any warranties or conditions
#       of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
#       PARTICULAR PURPOSE. You are solely responsible for determining the
#       appropriateness of using or redistributing the Work and assume any
#       risks associated with Your exercise of permissions under this License.
#
#    8. Limitation of Liability. In no event and under no legal theory,
#       whether in tort (including negligence), contract, or otherwise,
#       unless required by applicable law (such as deliberate and grossly
#       negligent acts) or agreed to in writing, shall any Contributor be
#       liable to You for damages, including any direct, indirect, special,
#       incidental, or consequential damages of any character arising as a
#       result of this License or out of the use or inability to use the
#       Work (including but not limited to damages for loss of goodwill,
#       work stoppage, computer failure or malfunction, or any and all
#       other commercial damages or losses), even if such Contributor
#       has been advised of the possibility of such damages.
#
#    9. Accepting Warranty or Additional Liability. While redistributing
#       the Work or Derivative Works thereof, You may choose to offer,
#       and charge a fee for, acceptance of support, warranty, indemnity,
#       or other liability obligations and/or rights consistent with this
#       License. However, in accepting such obligations, You may act only
#       on Your own behalf and on Your sole responsibility, not on behalf
#       of any other Contributor, and only if You agree to indemnify,
#       defend, and hold each Contributor harmless for any liability
#       incurred by, or claims asserted against, such Contributor by reason
#       of your accepting any such warranty or additional liability.
#
#    END OF TERMS AND CONDITIONS
#
#    Copyright 2021 Exathlon Project
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import logging
import os
from typing import List

import numpy as np
import pandas as pd

from .common import update_statistics_increment

_logger = logging.getLogger(__name__)

# spark streaming application ids
APP_IDS = tuple(range(1, 11))
# spark streaming trace types
TRACE_TYPES = (
    'undisturbed', 'bursty_input', 'bursty_input_crash',
    'stalled_input', 'cpu_contention', 'process_failure'
)
# anomaly types (the integer label for a type is its corresponding index in the list +1)
ANOMALY_TYPES = (
    'bursty_input', 'bursty_input_crash', 'stalled_input',
    'cpu_contention', 'driver_failure', 'executor_failure', 'unknown'
)


def extract_binary_ranges_ids(y):
    """Returns the start and (excluded) end indices of all contiguous ranges of 1s in the binary array `y`.

    Args:
        y (ndarray): 1d-array of binary elements.
    Returns:
        ndarray: array of `start, end` indices: `[[start_1, end_1], [start_2, end_2], ...]`.
    """
    y_diff = np.diff(y)
    start_ids = np.concatenate([[0] if y[0] == 1 else np.array([], dtype=int), np.where(y_diff == 1)[0] + 1])
    end_ids = np.concatenate([np.where(y_diff == -1)[0] + 1, [len(y)] if y[-1] == 1 else np.array([], dtype=int)])
    return np.array(list(zip(start_ids, end_ids)))


def load_trace(trace_path):
    """Loads a Spark trace as a pd.DataFrame from its full input path.

    Args:
        trace_path (str): full path of the trace to load (with file extension).
    Returns:
        pd.DataFrame: the trace indexed by time, with columns processed to be consistent between traces.
    """
    # load trace DataFrame with time as its converted datetime index
    trace_df = pd.read_csv(trace_path)
    trace_df.index = pd.to_datetime(trace_df['t'], unit='s')
    trace_df = trace_df.drop('t', axis=1)

    # remove the previous file prefix from streaming metrics for their name to be consistent across traces
    trace_df.columns = [
        c.replace(f'{"_".join(c.split("_")[1:10])}_', '') if 'StreamingMetrics' in c else c
        for c in trace_df.columns
    ]

    # return the DataFrame with sorted columns (they might be in a different order depending on the file)
    return trace_df.reindex(sorted(trace_df.columns), axis=1)


def load_labels(data_path: str) -> pd.DataFrame:
    _logger.info('loading ground-truth table...')
    labels = pd.read_csv(os.path.join(data_path, 'ground_truth.csv'))

    # convert timestamps to datetime format
    for c in ['root_cause_start', 'root_cause_end', 'extended_effect_end']:
        labels[c] = pd.to_datetime(labels[c], unit='s')
    _logger.info('done.')

    return labels

def load_raw_data(data_path: str, app_id: int = 0, trace_types: List[str] = TRACE_TYPES):
    """Spark-specific raw data loading.
    - The loaded periods as the application(s) traces.
    - The labels as the ground-truth table gathering events information for disturbed traces.
    - The periods information as lists initialized in the form `[file_name, trace_type]`.
    """
    period_dfs, periods_info = [], []

    # select relevant application keys (we also exclude 7 and 8 if all applications)
    app_keys = [f'app{app_id}'] if app_id != 0 else [f'app{i}' for i in set(APP_IDS) - {7, 8}]

    # load traces of the selected application(s) and type(s)
    for app_key in app_keys:
        _logger.info(f'loading traces of application {app_key.replace("app", "")}')
        app_path = os.path.join(data_path, app_key)
        file_names = os.listdir(app_path)
        for trace_type in trace_types:
            type_file_names = [
                fn for fn in file_names if int(fn.split('_')[1]) == TRACE_TYPES.index(trace_type)
            ]
            if len(type_file_names) > 0:
                _logger.info(f'loading {trace_type.replace("_", " ")} traces...')
                for fn in type_file_names:
                    period_dfs.append(load_trace(os.path.join(app_path, fn)))
                    periods_info.append([fn[:-4], trace_type])
                _logger.info('done.')
    assert len(period_dfs) > 0, 'no traces for the provided application(s) and type(s).'
    return period_dfs, periods_info


def add_anomaly_column(period_dfs, labels, periods_info, ignored_anomalies: str = 'none'):
    """Spark-specific `Anomaly` column extension.
    Note: `periods_info` is assumed to be of the form `[file_name, trace_type]`.
    `Anomaly` will be set to 0 if the record is outside any anomaly range, otherwise it will be
    set to another value depending on the range type (as defined by utils.spark.ANOMALY_TYPES).
    => The label for a given range type corresponds to its index in the ANOMALY_TYPES list +1.
    """
    _logger.info('adding an `Anomaly` column to the Spark traces...')
    for i, period_df in enumerate(period_dfs):
        period_df['Anomaly'] = 0
        file_name, trace_type = periods_info[i]
        if trace_type != 'undisturbed':
            for a_t in labels[labels['trace_name'] == file_name].itertuples():
                # ignore anomalies that had no impact on the recorded application if specified
                if not (ignored_anomalies == 'os.only' and a_t.anomaly_details == 'no_application_impact'):
                    a_start = a_t.root_cause_start
                    # either set the anomaly end to the root cause or extended effect end if the latter is set
                    a_end = a_t.root_cause_end if pd.isnull(a_t.extended_effect_end) else \
                        a_t.extended_effect_end
                    # set the label of an anomaly type as its index in the types list +1
                    period_df.loc[(period_df.index >= a_start) & (period_df.index <= a_end), 'Anomaly'] = \
                        ANOMALY_TYPES.index(a_t.anomaly_type) + 1
    _logger.info('done.')
    return period_dfs


def get_handled_nans(period_dfs):
    """Spark-specific handling of NaN values.
    The only NaN values that were recorded were for inactive executors, found equivalent
    of them being -1.
    """
    _logger.info('handling NaN values encountered in Spark traces...')
    for period_df in period_dfs:
        period_df.fillna(-1, inplace=True)
    _logger.info('done.')
    return period_dfs


def get_handled_executor_features(period_dfs):
    """Returns the period DataFrames with "handled" executor features.
    By looking at the features, we saw that some executor spots sometimes went to -1,
    presumably because we did not receive data from them within a given delay, without
    meaning the executors were not active anymore.
    The goal of this method is to detect such scenarios, in which case all -1 features are replaced
    with their last valid occurrence.
    Note: it was checked that if a given feature for an executor spot is -1, then all features
    from that spot are also -1, except for 1 or 2 records of some traces, which we argue is negligible.
    => To handle these very few cases, we would explicitly set all features to -1.

    Args:
        period_dfs (list): the list of input period DataFrames. Assumed without NaNs.
    Returns:
        list: the new period DataFrames, with handled executor features.
    """
    _logger.info('handling executor features in Spark traces...')
    # periods with handled executor features
    handled_dfs = [period_df.copy() for period_df in period_dfs]
    # copies used to extract continuous ranges of -1s
    extraction_dfs = [period_df.copy() for period_df in period_dfs]
    for handled_df, extraction_df in zip(handled_dfs, extraction_dfs):
        for executor_spot in range(1, 6):
            # take an arbitrary counter feature of the executor and extract -1 ranges for it
            counter_name = f'{executor_spot}_executor_runTime_count'
            extraction_df.loc[handled_df[counter_name] == -1, counter_name] = 1
            extraction_df.loc[handled_df[counter_name] != -1, counter_name] = 0
            for start_idx, end_idx in extract_binary_ranges_ids(extraction_df[counter_name].values):
                # only consider filling -1s if the range is between two non-(-1) ranges (end is excluded)
                if start_idx != 0 and end_idx != len(handled_df):
                    # if the counter was not reset, fill all executor features with their last valid value
                    preceding_counter = handled_df[counter_name].iloc[start_idx-1]
                    following_counter = handled_df[counter_name].iloc[end_idx]
                    if following_counter >= preceding_counter:
                        # end is included for the `loc` method
                        start_time, end_time = handled_df.index[start_idx], handled_df.index[end_idx-1]
                        valid_time = handled_df.index[start_idx-1]
                        for ft_name in [c for c in handled_df.columns if c[:2] == f'{executor_spot}_']:
                            handled_df.loc[start_time:end_time, ft_name] = handled_df.loc[valid_time, ft_name]
    _logger.info('done.')
    return handled_dfs


def get_handled_os_features(period_dfs):
    """Returns the period DataFrames with "handled" OS features.
    Similarly to some executor features, some OS features might happen to be -1 for no
    other reason than their real value not being sent fast enough by the monitoring software.
    This is here true for all encountered -1 values that do not span an entire trace.
    In such scenarios, we replace -1 features with their last valid value (or their next one
    if their last is not available).

    Args:
        period_dfs (list): the list of input period DataFrames. Assumed without NaNs.
    Returns:
        list: the new period DataFrames, with handled OS features.
    """
    _logger.info('handling OS features in Spark traces...')
    os_ft_names = [c for c in period_dfs[0].columns if c[:4] == 'node']
    handled_dfs = [period_df.copy() for period_df in period_dfs]
    for handled_df in handled_dfs:
        handled_df[os_ft_names] = handled_df[os_ft_names].replace(-1, np.nan).ffill().bfill().fillna(-1)
    _logger.info('done.')
    return handled_dfs


def handle_missing_values(period_dfs):
    """Returns the period DataFrames with "handled" OS features.
    Similarly to some executor features, some OS features might happen to be -1 for no
    other reason than their real value not being sent fast enough by the monitoring software.
    This is here true for all encountered -1 values that do not span an entire trace.
    In such scenarios, we replace -1 features with their last valid value (or their next one
    if their last is not available).

    Args:
        period_dfs (list): the list of input period DataFrames. Assumed without NaNs.
    Returns:
        list: the new period DataFrames, with handled OS features.
    """
    _logger.info('handling missing values in Spark traces...')
    for handled_df in period_dfs:
        handled_df[:] = handled_df.replace(-1, np.nan).ffill().bfill().fillna(-1)
    _logger.info('done.')
    return period_dfs


def add_executors_avg(period_df, original_treatment):
    """Adds executor features averaged across active executors, keeping or not the original ones.
    An executor is defined as "inactive" for a given feature if the value of the feature for
    this executor is -1.
    Note: it was checked that if a given feature for an executor spot is -1, then all features
    from that spot are also -1, except for 1 or 2 records of some traces, which we argue is negligible.
    => To handle these few cases, we should have set all the features to -1 (see
    `data.spark_manager.SparkManager.get_handled_executor_features`).
    """
    assert original_treatment in ['drop', 'keep'], 'original features treatment can only be `keep` or `drop`'

    # make sure to only try to average executor features
    exec_ft_names = [c[2:] for c in period_df.columns if c[:2] == '1_']
    # features groups to average across, each group of the form [`1_ft`, `2_ft`, ..., `5_ft`]
    avg_groups = [[c for c in period_df.columns if c[2:] == efn] for efn in exec_ft_names]

    # add features groups averaged across active executors to the result DataFrame
    averaged_df = pd.DataFrame()
    for group in avg_groups:
        # create `avg_ft` from [`1_ft`, `2_ft`, ..., `5_ft`]
        averaged_df = averaged_df.assign(
            **{f'avg_{group[0][2:]}': period_df[group].replace(-1, np.nan).mean(axis=1).fillna(-1)}
        )
    # prepend original input features if we choose to keep them
    if original_treatment == 'keep':
        averaged_df = pd.concat([period_df, averaged_df], axis=1)
    return averaged_df


def add_nodes_avg(period_df, original_treatment):
    """Adds node features averaged across nodes, keeping or not the original ones.
    """
    assert original_treatment in ['drop', 'keep'], 'original features treatment can only be `keep` or `drop`'

    # make sure to only try to average node features
    node_ft_names = [c[6:] for c in period_df.columns if c[:4] == 'node']
    # features groups to average across, each group of the form [`node5_ft`, `node6_ft`, ..., `node8_ft`]
    avg_groups = [[c for c in period_df.columns if c[6:] == nfn] for nfn in node_ft_names]

    # add features groups averaged across nodes to the result DataFrame
    averaged_df = pd.DataFrame()
    for group in avg_groups:
        # create `avg_node_ft` from [`node5_ft`, `node6_ft`, ..., `node8_ft`]
        averaged_df = averaged_df.assign(
            **{f'avg_node_{group[0][6:]}': period_df[group].mean(axis=1)}
        )
    # prepend original input features if we choose to keep them
    if original_treatment == 'keep':
        averaged_df = pd.concat([period_df, averaged_df], axis=1)
    return averaged_df


def add_differencing(period_df, diff_factor_str, original_treatment):
    """Adds features differences, either keeping or dropping the original ones.

    Args:
        period_df (pd.DataFrame): input period DataFrame.
        diff_factor_str (str): differencing factor as a string integer.
        original_treatment (str): either `keep` or `drop`, specifying what to do with original features.
    Returns:
        pd.DataFrame: the input DataFrame with differenced features, with or without the original ones.
    """
    assert original_treatment in ['drop', 'keep'], 'original features treatment can only be `keep` or `drop`'
    # apply differencing and drop records with NaN values
    difference_df = period_df.diff(int(diff_factor_str)).dropna()
    difference_df.columns = [f'{diff_factor_str}_diff_{c}' for c in difference_df.columns]
    # prepend original input features if we choose to keep them (implicit join if different counts)
    if original_treatment == 'keep':
        difference_df = pd.concat([period_df, difference_df], axis=1)
    return difference_df


def get_resampled(period_dfs, sampling_period, agg='mean', anomaly_col=False, pre_sampling_period=None):
    """
    Returns the period DataFrames resampled to `sampling_period` using the provided aggregation function.
    If `anomaly_col` is False, periods indices will also be reset to start from a round date,
    in order to keep them aligned with any subsequently resampled labels.

    Args:
        period_dfs (list): the period DataFrames to resample.
        sampling_period (str): the new sampling period, as a valid argument to `pd.Timedelta`.
        agg (str): the aggregation function defining the resampling (e.g. `mean`, `median` or `max`).
        anomaly_col (bool): whether the provided DataFrames have an `Anomaly` column, to resample differently.
        pre_sampling_period (str|None): original sampling period of the DataFrames.
            If None and `anomaly_col` is False, the original sampling period will be inferred
            from the first two records of the first DataFrame.
    Returns:
        list: the same periods with resampled records.
    """
    resampled_dfs, sampling_p, pre_sampling_p = [], pd.Timedelta(sampling_period), None
    feature_cols = [c for c in period_dfs[0].columns if c != 'Anomaly']
    if not anomaly_col:
        # turn any original sampling period to type `pd.Timedelta`
        pre_sampling_p = pd.Timedelta(np.diff(period_dfs[0].index[:2])[0]) \
            if pre_sampling_period is None else pd.Timedelta(pre_sampling_period)
    _logger.info(f'resampling periods applying records `{agg}` every {sampling_period}...')
    for df in period_dfs:
        if not anomaly_col:
            # reindex the period DataFrame to start from a round date before downsampling
            df = df.set_index(pd.date_range('01-01-2000', periods=len(df), freq=pre_sampling_p))
        resampled_df = df[feature_cols].resample(sampling_p).agg(agg).ffill().bfill()
        if anomaly_col:
            # if no records during `sampling_p`, we choose to simply repeat the label of the last one here
            resampled_df['Anomaly'] = df['Anomaly'].resample(sampling_p).agg('max').ffill().bfill()
        resampled_dfs.append(resampled_df)
    _logger.info('done.')
    return resampled_dfs


def preprocess_exathlon_data(raw_data_dir: str, out_data_dir: str, app_ids: List[int] = set(APP_IDS) - {7,8},
                             subsample: str = None):
    """
    Preprocess Exathlon dataset for experiments

    :param raw_data_dir: Path to raw data folder in the dataset
    :param out_data_dir: Folder to output the processed data
    :param app_ids: Which all app data should be processed
    :param subsample: The new sampling period for DataFrames. Should be a valid argument to :class:`pandas.Timedelta`
    """
    labels = load_labels(raw_data_dir)

    for app_id in app_ids:
        period_dfs, periods_info = load_raw_data(raw_data_dir, app_id=app_id)
        # add an `Anomaly` column to the period DataFrames based on the labels
        period_dfs = add_anomaly_column(period_dfs, labels, periods_info)
        # handle NaN values in the raw period DataFrames
        period_dfs = get_handled_nans(period_dfs)
        # resample periods with their original resolution to avoid duplicate indices (max to remove the effect of -1s)
        period_dfs = get_resampled(period_dfs, '1s', agg='max', anomaly_col=True)

        # handle any -1 executor and OS values that unexpectedly occurred during monitoring
        # period_dfs = get_handled_executor_features(period_dfs)
        # period_dfs = get_handled_os_features(period_dfs)
        period_dfs = handle_missing_values(period_dfs)
        # apply the specified period pruning if relevant
        # period_dfs = get_pruned_periods(period_dfs)

        # resample period DataFrames to the new sampling period if different from the original
        if subsample is not None:
            period_dfs = get_resampled(period_dfs, subsample, anomaly_col=True)

        # Project and transform features
        original_features = [
            'driver_StreamingMetrics_streaming_lastCompletedBatch_processingDelay_value',
            'driver_StreamingMetrics_streaming_lastCompletedBatch_schedulingDelay_value',
            'driver_StreamingMetrics_streaming_lastCompletedBatch_totalDelay_value',
            'Anomaly'
        ]
        differencing_features = [
            'driver_StreamingMetrics_streaming_totalCompletedBatches_value',
            'driver_StreamingMetrics_streaming_totalProcessedRecords_value',
            'driver_StreamingMetrics_streaming_totalReceivedRecords_value',
            'driver_StreamingMetrics_streaming_lastReceivedBatch_records_value',
            'driver_BlockManager_memory_memUsed_MB_value',
            'driver_jvm_heap_used_value',
            *[f'node{i}_CPU_ALL_Idle%' for i in range(5, 9)]
        ]
        avg_differencing_features = [
            *[f'{i}_executor_filesystem_hdfs_write_ops_value' for i in range(1, 6)],
            *[f'{i}_executor_cpuTime_count' for i in range(1, 6)],
            *[f'{i}_executor_runTime_count' for i in range(1, 6)],
            *[f'{i}_executor_shuffleRecordsRead_count' for i in range(1, 6)],
            *[f'{i}_executor_shuffleRecordsWritten_count' for i in range(1, 6)],
            *[f'{i}_jvm_heap_used_value' for i in range(1, 6)]
        ]
        for i, df in enumerate(period_dfs):
            # Remove first row, because it is removed during differencing as well
            orig_df = df[original_features].iloc[1:]
            diff_df = add_differencing(df[differencing_features], '1', 'drop')
            avg_df = add_executors_avg(df[avg_differencing_features], 'drop')
            avg_df = add_differencing(avg_df, '1', 'drop')
            period_dfs[i] = pd.concat([orig_df, diff_df, avg_df], axis=1)

        os.makedirs(os.path.join(out_data_dir, 'train'), exist_ok=True)
        os.makedirs(os.path.join(out_data_dir, 'test'), exist_ok=True)

        mean, min, max, n = None, None, None, 0
        for df, info in zip(period_dfs, periods_info):
            type = 'train' if info[1] == 'undisturbed' else 'test'
            df.to_csv(os.path.join(out_data_dir, type, f'{info[0]}.csv'))

            # Update statistics
            if type == 'train':
                mean, min, max, n = update_statistics_increment(df.loc[:, df.columns != 'Anomaly'], mean, min, max, n)

        # Save dataset statistics
        stats_file = os.path.join(out_data_dir, 'train', f'train_stats_{app_id}.npz')
        np.savez(stats_file, mean=mean, min=min, max=max)
