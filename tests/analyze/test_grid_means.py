"""Tests for wem.analyze.grid_means module imports and mean_from_quantiles."""

import numpy as np
import pandas as pd
import pytest

from wem.constants import QCOLS


class TestGridMeansModule:
    def test_import_grid_means(self):
        import wem.analyze.grid_means  # noqa: F401

    def test_main_exists(self):
        from wem.analyze import grid_means

        assert callable(grid_means.main)

    def test_qcols_imported(self):
        assert len(QCOLS) == 101

    def test_mean_from_quantiles_uniform(self):
        from wem.utils.quantiles import mean_from_quantiles

        # A single row where every quantile column = 5.0
        # The trapezoidal integral of a constant c over [0,1] equals c.
        row = {col: 5.0 for col in QCOLS}
        df = pd.DataFrame([row])
        result = mean_from_quantiles(df)
        np.testing.assert_allclose(result, 5.0, atol=1e-4)
