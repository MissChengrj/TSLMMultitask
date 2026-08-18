import numpy as np

from tslm_multitask.data.masks import missing_indices_from_mask, observed_mask_from_values


def test_zero_value_not_treated_as_missing():
    values = np.array([1.0, 0.0, np.nan, -2.0], dtype=np.float32)

    observed_mask = observed_mask_from_values(values)

    assert observed_mask.tolist() == [True, True, False, True]
    assert missing_indices_from_mask(observed_mask).tolist() == [2]
