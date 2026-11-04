"""Scientific implementation checks: masks and bag ordering must not change predictions."""

import unittest
import torch
from bag_model import AttentionHead


class BagInvariants(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.model = AttentionHead(d=12).eval()
        self.x = torch.randn(3, 4, 12)
        self.mask = torch.tensor(
            [[True, True, False, False], [True] * 4, [True, False, False, False]]
        )

    def test_padding_cannot_change_predictions(self):
        changed = self.x.clone()
        changed[~self.mask] = 100000
        a, weights = self.model(self.x, self.mask)
        b, _ = self.model(changed, self.mask)
        torch.testing.assert_close(a, b)
        self.assertTrue(
            torch.all(weights.masked_select(~self.mask[:, None, :].expand_as(weights)) == 0)
        )
        torch.testing.assert_close(weights.sum(-1), torch.ones(3, 7))

    def test_bag_permutation_equivariance(self):
        order = torch.tensor([2, 0, 3, 1])
        a, weights = self.model(self.x, self.mask)
        b, moved = self.model(self.x[:, order], self.mask[:, order])
        torch.testing.assert_close(a, b)
        torch.testing.assert_close(weights[:, :, order], moved)

    def test_one_chunk_has_unit_attention(self):
        _, weights = self.model(self.x, self.mask)
        torch.testing.assert_close(weights[2, :, 0], torch.ones(7))


if __name__ == "__main__":
    unittest.main(verbosity=2)
