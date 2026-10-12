import numpy as np

from topopt_utils.fea.spatial_interactions import (
    compute_distance_bin_stats,
    directional_coupling_stats,
    distance_conditioned_directional_coupling,
    filter_support_geometry,
    interaction_energy_tail,
    interaction_energy_tail_curve,
    pairwise_distances,
)


def test_distance_bins_and_masking_for_active_pairs():
    x = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    y = np.zeros_like(x)
    P = np.array(
        [
            [1.0, 0.8, 0.3],
            [0.8, 1.0, 0.9],
            [0.3, 0.9, 1.0],
        ],
        dtype=np.float64,
    )
    R = P.copy()
    active_mask = np.array([True, True, True], dtype=bool)
    edges = np.array([0.0, 0.5, 1.5, 3.0], dtype=np.float64)

    stats = compute_distance_bin_stats(P, R, active_mask, x, y, edges)

    assert stats["num_active_pairs"] == 3
    assert [item["pair_count"] for item in stats["pair_statistics"]] == [0, 2, 1]
    assert np.isclose(stats["pair_statistics"][1]["mean_abs_R"], 0.85)
    assert np.isclose(stats["pair_statistics"][2]["mean_abs_R"], 0.3)

    active_mask[1] = False
    stats_masked = compute_distance_bin_stats(P, R, active_mask, x, y, edges)
    assert stats_masked["num_active_pairs"] == 1
    assert stats_masked["pair_statistics"][2]["pair_count"] == 1

    d = pairwise_distances(x, y)
    np.testing.assert_allclose(d, np.array([1.0, 2.0, 1.0], dtype=np.float64))


def test_synthetic_example_has_expected_directional_classes():
    x = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float64)
    y = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
    active_mask = np.ones(4, dtype=bool)
    R = np.ones((4, 4), dtype=np.float64)
    stats = directional_coupling_stats(R, active_mask, x, y)

    assert stats["horizontal"]["pair_count"] == 2
    assert stats["vertical"]["pair_count"] == 2
    assert stats["diagonal"]["pair_count"] == 2
    assert np.isclose(stats["horizontal"]["mean_abs_R"], 1.0)
    assert np.isclose(stats["diagonal"]["mean_abs_R"], 1.0)
    assert np.isclose(stats["vertical"]["mean_abs_R"], 1.0)


def test_energy_tail_monotonic_and_zero_denominator_handling():
    x = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    y = np.zeros_like(x)
    P = np.array(
        [
            [1.0, 0.8, 0.3],
            [0.8, 1.0, 0.9],
            [0.3, 0.9, 1.0],
        ],
        dtype=np.float64,
    )
    cutoffs = np.array([0.0, 0.5, 1.0, 1.5, 5.0], dtype=np.float64)
    curve = interaction_energy_tail_curve(P, x, y, cutoffs)
    vals = [item["tail_fraction"] for item in curve["tail_values"]]
    assert vals[-1] == 0.0
    assert np.all(np.diff(vals) <= 1e-12)
    assert np.isclose(vals[0], 1.0)

    zero = interaction_energy_tail(np.eye(3), x, y, 0.5)
    assert zero == 0.0


def test_empty_bins_and_signed_statistics():
    x = np.array([0.0, 2.0], dtype=np.float64)
    y = np.array([0.0, 0.0], dtype=np.float64)
    P = np.array([[1.0, 0.4], [0.4, 1.0]], dtype=np.float64)
    R = np.array([[1.0, -0.6], [-0.6, 1.0]], dtype=np.float64)
    edges = np.array([0.0, 0.5, 1.5, 3.0], dtype=np.float64)
    stats = compute_distance_bin_stats(P, R, np.array([True, True]), x, y, edges)

    assert stats["pair_statistics"][0]["pair_count"] == 0
    assert np.isnan(stats["pair_statistics"][0]["mean_abs_R"])
    assert stats["pair_statistics"][2]["pair_count"] == 1
    assert np.isclose(stats["pair_statistics"][2]["mean_signed_R"], -0.6)
    assert np.isclose(stats["pair_statistics"][2]["fraction_positive_R"], 0.0)
    assert np.isclose(stats["pair_statistics"][2]["fraction_negative_R"], 1.0)

    stats_dir = distance_conditioned_directional_coupling(
        R,
        np.array([True, True]),
        x,
        y,
        edges,
        min_pairs=2,
    )
    assert stats_dir["bins"][0]["bin_status"] == "empty_bin"
    assert stats_dir["bins"][2]["bin_status"] == "below_min_pairs"
    assert stats_dir["bins"][2]["support_ok"] is False


def test_permutation_invariance_of_pairwise_statistics():
    x = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    y = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    P = np.array(
        [
            [1.0, 0.6, 0.2],
            [0.6, 1.0, 0.7],
            [0.2, 0.7, 1.0],
        ],
        dtype=np.float64,
    )
    R = P.copy()
    active_mask = np.array([True, True, True], dtype=bool)
    edges = np.array([0.0, 0.5, 1.5, 3.0], dtype=np.float64)

    stats_a = compute_distance_bin_stats(P, R, active_mask, x, y, edges)
    perm = np.array([2, 0, 1], dtype=int)
    stats_b = compute_distance_bin_stats(P[np.ix_(perm, perm)], R[np.ix_(perm, perm)], active_mask[perm], x[perm], y[perm], edges)

    assert [bin_info["pair_count"] for bin_info in stats_a["pair_statistics"]] == [
        bin_info["pair_count"] for bin_info in stats_b["pair_statistics"]
    ]
    assert np.allclose(
        [bin_info["mean_abs_R"] for bin_info in stats_a["pair_statistics"]],
        [bin_info["mean_abs_R"] for bin_info in stats_b["pair_statistics"]],
        equal_nan=True,
    )


def test_physical_distance_and_active_pair_tail():
    x = np.array([0.0, 2.0, 0.0], dtype=np.float64)
    y = np.array([0.0, 0.0, 2.0], dtype=np.float64)
    d = pairwise_distances(x, y)
    assert np.allclose(np.sort(d), np.array([2.0, 2.0, 2.8284271247461903]))

    P = np.array(
        [
            [1.0, 0.4, 0.2],
            [0.4, 1.0, 0.5],
            [0.2, 0.5, 1.0],
        ],
        dtype=np.float64,
    )
    active_mask = np.array([True, True, False], dtype=bool)
    tail = interaction_energy_tail(P, x, y, 2.0, active_mask=active_mask)
    assert 0.0 <= tail <= 1.0

    exact = interaction_energy_tail(P, x, y, np.max(pairwise_distances(x, y)), active_mask=None)
    assert exact == 0.0


def test_filter_support_geometry_and_distance_binned_directional_support():
    geom = filter_support_geometry(2.0, dx=1.0, dy=1.0)
    assert np.isclose(geom["rmin_element_count"], 2.0)
    assert np.isclose(geom["rmin_physical_x"], 2.0)
    assert np.isclose(geom["rmin_physical_y"], 2.0)
    assert np.isclose(geom["rmin_physical_diagonal"], 2.0 * np.sqrt(2.0))

    x = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    y = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    R = np.ones((4, 4), dtype=np.float64)
    active_mask = np.ones(4, dtype=bool)
    edges = np.array([0.0, 1.0, 2.0, 4.0], dtype=np.float64)
    stats = distance_conditioned_directional_coupling(R, active_mask, x, y, edges, min_pairs=2)

    assert stats["bins"][0]["support_ok"] is False
    assert stats["bins"][2]["support_ok"] is True
    assert stats["bins"][2]["horizontal"]["pair_count"] == 3
    assert stats["bins"][2]["vertical"]["pair_count"] == 0
    assert stats["bins"][2]["diagonal"]["pair_count"] == 0
