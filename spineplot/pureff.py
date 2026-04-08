import numpy as np
from scipy.stats import binom
import pandas as pd
import matplotlib.pyplot as plt

from artists import SpineArtist
from efficiency import SpineEfficiency
from purity import SpinePurity
from utilities import mark_pot, mark_preliminary


class SpineEfficiencyPurity(SpineArtist):
    """
    Combined efficiency + purity table artist. Delegates all calculation
    to SpineEfficiency and SpinePurity internally, then merges the
    results into a single four-column table:
        Cut | Diff. Eff | Cum. Eff | Diff. Pur | Cum. Pur
    """

    def __init__(self, variable, categories, cuts, title,
                xrange=None, xtitle=None, show_option='table', npts=1e6,
                all_categories=None):
        super().__init__(title)
        self._variable = variable
        self._categories = categories
        self._cuts = cuts
        self._show_option = show_option
        self._efficiency = SpineEfficiency(variable, categories, cuts, title,
                                        xrange, xtitle, show_option, npts)
        purity_cats = all_categories if all_categories is not None else categories
        self._purity = SpinePurity(variable, purity_cats, cuts, title,
                                xrange, xtitle, show_option, npts,
                                signal_categories=categories)

    def add_sample(self, sample, is_ordinate):
        super().add_sample(sample, is_ordinate)
        tree_name = getattr(sample, '_tree_name', None)
        
        if tree_name == 'signal':
            # Only efficiency uses the signal tree (true denominator)
            self._efficiency.add_sample(sample, is_ordinate)
        elif tree_name == 'selected':
            # Only purity uses the selected tree (all categories for denominator)
            self._purity.add_sample(sample, is_ordinate)
        else:
            # Fallback: feed both (single-tree config)
            self._efficiency.add_sample(sample, is_ordinate)
            self._purity.add_sample(sample, is_ordinate)

    def draw(self, ax, show_option='table', percentage=True,
         show_seqeff=True, show_unseqeff=True,
         show_seqpur=True, show_unseqpur=True,
         yrange=None, npts=1e6, style=None,
         logx=False, logy=False):

        ax.set_title(self._title)

        groups = []
        for category in self._categories.values():
            if category not in groups:
                groups.append(category)

        if show_option == 'table':
            if percentage:
                eff_fmt      = lambda x, y, z: rf'${100*x:.2f}^{{\ +{100*y:.2f}}}_{{\ -{100*z:.2f}}}$'
                pur_fmt      = lambda x: rf'${100*x:.2f}$'
                DIFF_PUR     = 'Diff.\nPur. [%]'
                CUMU_PUR     = 'Cum.\nPur. [%]'
                DIFF_EFF     = 'Diff.\nEff. [%]'
                CUMU_EFF     = 'Cum.\nEff. [%]'
            else:
                eff_fmt      = lambda x, y, z: rf'${x:.4f}^{{\ +{y:.2e}}}_{{\ -{z:.2e}}}$'
                pur_fmt      = lambda x: rf'${x:.4f}$'
                DIFF_PUR     = 'Diff.\nPur.'
                CUMU_PUR     = 'Cum.\nPur.'
                DIFF_EFF     = 'Diff.\nEff.'
                CUMU_EFF     = 'Cum.\nEff.'

            ax.axis('off')

            # Build the full results DataFrame with ALL four columns first,
            # then filter to only the requested ones at the end.
            all_rows = []
            group_endpoint = {}

            for group in groups:
                _, eff_cv, eff_ms, eff_ps = self._efficiency.reduce(group, significance=0.6827)
                _, pur_cv, pur_ms, pur_ps = self._purity.reduce(group, significance=0.6827)

                seq_eff   = [v for k, v in eff_cv.items() if 'unbinned_seq_'   in k and 'unseq' not in k]
                unseq_eff = [v for k, v in eff_cv.items() if 'unbinned_unseq_' in k]
                seq_eff_ms   = [v for k, v in eff_ms.items() if 'unbinned_seq_'   in k and 'unseq' not in k]
                unseq_eff_ms = [v for k, v in eff_ms.items() if 'unbinned_unseq_' in k]
                seq_eff_ps   = [v for k, v in eff_ps.items() if 'unbinned_seq_'   in k and 'unseq' not in k]
                unseq_eff_ps = [v for k, v in eff_ps.items() if 'unbinned_unseq_' in k]

                seq_pur   = [v for k, v in pur_cv.items() if 'unbinned_seq_'   in k and 'unseq' not in k]
                unseq_pur = [v for k, v in pur_cv.items() if 'unbinned_unseq_' in k]

                cut_labels = list(self._cuts.values())
                for ci, cut_label in enumerate(cut_labels):
                    all_rows.append({
                        r'   ':    group if ci == 0 else '',
                        r'Cut':    cut_label,
                        DIFF_PUR:  pur_fmt(unseq_pur[ci]),
                        CUMU_PUR:  pur_fmt(seq_pur[ci]),
                        DIFF_EFF:  eff_fmt(unseq_eff[ci], unseq_eff_ms[ci], unseq_eff_ps[ci]),
                        CUMU_EFF:  eff_fmt(seq_eff[ci],   seq_eff_ms[ci],   seq_eff_ps[ci]),
                    })
                group_endpoint[group] = len(all_rows)

            results = pd.DataFrame(all_rows)

            # Select only the columns the user wants
            col_names = [r'   ', r'Cut'] if len(groups) > 1 else [r'Cut']
            if show_unseqpur:
                col_names.append(DIFF_PUR)
            if show_seqpur:
                col_names.append(CUMU_PUR)
            if show_unseqeff:
                col_names.append(DIFF_EFF)
            if show_seqeff:
                col_names.append(CUMU_EFF)
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

            def calc_bbox_yext(obj):
                figure = plt.gcf()
                bbox = obj.get_window_extent(renderer=figure.canvas.get_renderer())
                p0, p1 = figure.transFigure.inverted().transform(bbox)
                return p1[1] - p0[1]

            scale = 2.75
            while calc_bbox_yext(table) > 0.92:
                table.scale(1, 1 / scale)
                scale -= 0.05
                table.scale(1, scale)
            if scale < 2.0:
                print(f'Warning: Table `{self._title}` is too large (scale={scale:.2f}). '
                    f'Consider a taller figsize.')

            if style.mark_pot:
                mark_pot(ax, self._exposure, style.mark_pot_horizontal, vadj=0.1)
            if style.mark_preliminary is not None:
                mark_preliminary(ax, style.mark_preliminary, vadj=0.1)