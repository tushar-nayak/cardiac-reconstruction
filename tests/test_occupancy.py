import torch
import unittest

from src_code.src.occupancy import evaluate_gaussian_occupancy
from src_code.src.models.gaussian_model import GaussianModel


class OccupancyTests(unittest.TestCase):
  def test_model_scales_remain_finite_when_raw_parameter_is_negative(self):
    model = GaussianModel(num_gaussians=2, latent_dim=4)
    with torch.no_grad():
      model.scales.fill_(-100.0)
    params = model(torch.zeros(1, 4))
    self.assertTrue(torch.isfinite(params["scales"]).all())

  def test_occupancy_is_bounded_and_monotonic_in_opacity(self):
    points = torch.tensor([[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]])
    means = torch.zeros(1, 1, 3)
    scales = torch.ones(1, 1, 3)
    low = evaluate_gaussian_occupancy(points, means, scales, torch.tensor([[[0.2]]]))
    high = evaluate_gaussian_occupancy(points, means, scales, torch.tensor([[[0.8]]]))

    self.assertTrue(torch.all((low >= 0) & (low <= 1)))
    self.assertTrue(torch.all(high >= low))
    self.assertGreater(high[0, 0], high[0, 1])


  def test_chunking_matches_single_pass(self):
    torch.manual_seed(3)
    points = torch.randn(2, 23, 3)
    means = torch.randn(2, 7, 3)
    scales = torch.rand(2, 7, 3) + 0.2
    opacities = torch.rand(2, 7, 1)
    full = evaluate_gaussian_occupancy(points, means, scales, opacities, chunk_size=100)
    chunked = evaluate_gaussian_occupancy(points, means, scales, opacities, chunk_size=5)
    self.assertTrue(torch.allclose(full, chunked, atol=1e-6))
