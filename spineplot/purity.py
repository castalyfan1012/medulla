import numpy as np
from scipy.stats import binom
import pandas as pd
import matplotlib.pyplot as plt

from artists import SpineArtist
from style import Style
from variable import Variable
from utilities import mark_pot, mark_preliminary

class SpinePurity(SpineArtist):
    """
    Purity of the selection. For a given group (e.g. signal), purity
    is defined as:
        numerator   = weighted events from that group passing the cut
        denominator = weighted events from ALL groups passing the cut
    Uses the same Bayesian binomial posterior approach as SpineEfficiency.
    """

    def __init__(self, variable, categories, cuts, title,
                 xrange=None, xtitle=None, show_option='table',
                 npts=1e6, signal_categories=None):
        super().__init__(title)
        self._variable = variable
        self._samples = list()
        self._categories = categories        # FULL categories (all backgrounds)
        self._signal_categories = signal_categories if signal_categories is not None else categories
        self._cuts = cuts
        self._title = title
        self._xrange = xrange
        self._xtitle = xtitle
        self._show_option = show_option
        self._npts = int(npts)
        # Per-group numerator posteriors (signal events passing cut)
        self._posteriors = dict()
        # Global denominator: total weighted events passing cut (all groups)
        self._global_totals = dict()

    def add_sample(self, sample, is_ordinate):
        super().add_sample(sample, is_ordinate)
        self._samples.append(sample)
        self.calculate(sample)

    @staticmethod
    def multiply_posteriors(pos0, pos1):
        pos = pos0 * pos1
        if len(pos.shape) == 1:
            total = np.sum(pos)
            if total > 0:
                pos /= total
        return pos

    def calculate(self, sample):
        purities = np.linspace(0.0, 1.0, self._npts)

        data, weights = sample.get_data(
            [self._variable._key, *self._cuts.keys()],
            with_mask=self._variable.mask
        )

        # --- First pass: accumulate global weighted totals across ALL categories ---
        for category, values in data.items():
            if category not in self._categories:
                continue

            w = weights[category]

            # Initialize global denominator storage once
            if 'global' not in self._global_totals:
                self._global_totals['global'] = {
                    f'seq_{c}':   0.0 for c in self._cuts.keys()
                }
                self._global_totals['global'].update({
                    f'unseq_{c}': 0.0 for c in self._cuts.keys()
                })

            for ci, (cut, _) in enumerate(self._cuts.items()):
                seq_mask   = np.all(values[1:ci+2], axis=0)
                unseq_mask = values[ci+1].to_numpy(bool)
                self._global_totals['global'][f'seq_{cut}']   += float(np.sum(w[seq_mask]))
                self._global_totals['global'][f'unseq_{cut}'] += float(np.sum(w[unseq_mask]))

        # --- Second pass: accumulate per-group numerator posteriors ---
        for category, values in data.items():
            if category not in self._categories:
                continue

            group = self._categories[category]
            w = weights[category]

            # Initialize per-group posterior storage
            if group not in self._posteriors:
                self._posteriors[group] = {
                    f'unbinned_seq_{c}':   np.ones(purities.shape)
                    for c in self._cuts.keys()
                }
                self._posteriors[group].update({
                    f'unbinned_unseq_{c}': np.ones(purities.shape)
                    for c in self._cuts.keys()
                })

            for ci, (cut, _) in enumerate(self._cuts.items()):
                seq_mask   = np.all(values[1:ci+2], axis=0)
                unseq_mask = values[ci+1].to_numpy(bool)

                # Weighted signal (numerator) and weighted total (denominator)
                s_seq   = float(np.sum(w[seq_mask]))
                s_unseq = float(np.sum(w[unseq_mask]))
                n_seq   = self._global_totals['global'][f'seq_{cut}']
                n_unseq = self._global_totals['global'][f'unseq_{cut}']

                # Round to nearest integer for binom.pmf (it needs integer counts)
                # Scale to preserve ratio: use effective counts
                if n_seq > 0:
                    eff_s_seq = round(s_seq)
                    eff_n_seq = round(n_seq)
                    self._posteriors[group][f'unbinned_seq_{cut}'] = SpinePurity.multiply_posteriors(
                        self._posteriors[group][f'unbinned_seq_{cut}'],
                        binom.pmf(eff_s_seq, eff_n_seq, purities)
                    )

                if n_unseq > 0:
                    eff_s_unseq = round(s_unseq)
                    eff_n_unseq = round(n_unseq)
                    self._posteriors[group][f'unbinned_unseq_{cut}'] = SpinePurity.multiply_posteriors(
                        self._posteriors[group][f'unbinned_unseq_{cut}'],
                        binom.pmf(eff_s_unseq, eff_n_unseq, purities)
                    )

    def reduce(self, group, significance=0.6827):
        purities = np.linspace(0.0, 1.0, self._npts)

        if group not in self._posteriors:
            raise ValueError(f"Group '{group}' not in purity posteriors.")

        sig = [0.5 - significance / 2.0, 0.5 + significance / 2.0]
        cv, msigma, psigma = {}, {}, {}

        for key, posterior in self._posteriors[group].items():
            total = np.sum(posterior)
            if total <= 0:
                cv[key], msigma[key], psigma[key] = 0.0, 0.0, 0.0
                continue

            posterior = posterior / total
            cv[key]        = purities[int(np.argmax(posterior))]
            cumulative     = np.cumsum(posterior)
            msigma[key]    = max(0.0, cv[key] - purities[int(np.argmax(cumulative > sig[0]))])
            psigma[key]    = max(0.0, purities[int(np.argmax(cumulative > sig[1]))] - cv[key])

        return None, cv, msigma, psigma

    def draw(self, ax, show_option, percentage=True, show_seqpur=True,
             show_unseqpur=True, yrange=None, style=None, logx=False, logy=False):

        ax.set_title(self._title)

        groups = []
        for category in self._signal_categories.values():
            if category not in groups:
                groups.append(category)

        if show_option == 'table':
            if percentage:
                formatter = lambda x: rf'${100*x:.2f}$'
                diff_key  = 'Differential\nPurity [%]'
                cumu_key  = 'Cumulative\nPurity [%]'
            else:
                formatter = lambda x: rf'${x:.4f}$'
                diff_key  = 'Differential\nPurity'
                cumu_key  = 'Cumulative\nPurity'

            ax.axis('off')
            results = pd.DataFrame()
            group_endpoint = {}

            for group in groups:
                _, cv, msigma, psigma = self.reduce(group, significance=0.6827)

                seq   = lambda d: [v for k, v in d.items() if 'unbinned_seq_'   in k and 'unseq' not in k]
                unseq = lambda d: [v for k, v in d.items() if 'unbinned_unseq_' in k]

                entry = {
                    r'   ':   [group] + [r'' for _ in range(1, len(self._cuts))],
                    r'Cut':   list(self._cuts.values()),
                    diff_key: [formatter(x) for x in unseq(cv)],
                    cumu_key: [formatter(x) for x in seq(cv)],
                }
                results = pd.concat([results, pd.DataFrame(entry)])
                group_endpoint[group] = len(results)

            col_names = [r'   ', r'Cut'] if len(groups) > 1 else [r'Cut']
            if show_unseqpur:
                col_names.append(diff_key)
            if show_seqpur:
                col_names.append(cumu_key)
            results = results[col_names]

            table_data = [results.columns.to_list()] + results.values.tolist()
            table = ax.table(cellText=table_data, colLabels=None,
                             loc='center', cellLoc='center', edges='T')
            table.scale(1, 2.75)

            for i in range(2, len(table_data)):
                if i == len(table_data) - 1:
                    for j in range(len(table_data[i])):
                        table[i, j].visible_edges = 'B'
                else:
                    for j in range(len(table_data[i])):
                        table[i, j].visible_edges = 'open'
                    if i in group_endpoint.values():
                        table[i, 0].visible_edges = 'B'

            if style.mark_pot:
                mark_pot(ax, self._exposure, style.mark_pot_horizontal, vadj=0.1)
            if style.mark_preliminary is not None:
                mark_preliminary(ax, style.mark_preliminary, vadj=0.1)