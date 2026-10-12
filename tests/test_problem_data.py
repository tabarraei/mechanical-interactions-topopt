import torch

from topo_transformer import problem_data as pd


def test_sample_load_count_is_uniform_across_all_load_counts():
    generator = torch.Generator()
    generator.manual_seed(7)

    draws = [pd._sample_load_count(generator, max_loads=4, p_two_loads=0.0) for _ in range(32)]

    assert set(draws) == {1, 2, 3, 4}, draws
