import math
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from scripts.segmentation_pipeline import (
    CommonUNetDecoder,
    DOFADenseFeatureAdapter,
    FoundationSegmentationModel,
    PanopticonDenseFeatureAdapter,
    SegmentationMetricAccumulator,
    audit_segmentation_model,
    collect_segmentation_predictions,
    export_segmentation_predictions,
    segmentation_loss,
    segmentation_metrics,
    validate_segmentation_prediction_export,
)
from scripts.mc_dropout import activate_downstream_mc_dropout


class FakeDOFAPatchEmbed(nn.Module):
    grid_size = (2, 2)

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 4)
        self.last_waves = None

    def forward(self, image, waves):
        self.last_waves = waves.detach().clone()
        pooled = torch.nn.functional.adaptive_avg_pool2d(image, (2, 2))
        tokens = pooled.flatten(2).transpose(1, 2)
        return self.projection(tokens), None


class FakeDOFABackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = FakeDOFAPatchEmbed()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 4))
        self.pos_embed = nn.Parameter(torch.zeros(1, 5, 4), requires_grad=False)
        self.blocks = nn.ModuleList([nn.Identity()])
        self.norm = nn.LayerNorm(4)


class FakePanopticonModel(nn.Module):
    num_prefix_tokens = 1
    num_features = 4

    def __init__(self):
        super().__init__()
        self.patch_embed = type("PatchMetadata", (), {"grid_size": (2, 2)})()
        self.projection = nn.Linear(3, 4)
        self.last_channel_ids = None

    def forward_features(self, payload):
        self.last_channel_ids = payload["chn_ids"].detach().clone()
        pooled = torch.nn.functional.adaptive_avg_pool2d(payload["imgs"], (2, 2))
        tokens = self.projection(pooled.flatten(2).transpose(1, 2))
        return torch.cat((torch.zeros(tokens.shape[0], 1, 4, device=tokens.device), tokens), dim=1)


class FakePanopticonBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = FakePanopticonModel()


class ToyDenseAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Conv2d(3, 4, kernel_size=1)

    def forward(self, image):
        return torch.nn.functional.adaptive_avg_pool2d(self.backbone(image), (2, 2))


class SegmentationPipelineTests(unittest.TestCase):
    def test_model_specific_adapters_emit_same_dense_contract(self):
        image = torch.randn(2, 3, 8, 8)
        dofa = DOFADenseFeatureAdapter(FakeDOFABackbone(), [0.665, 0.560, 0.490])
        panopticon = PanopticonDenseFeatureAdapter(FakePanopticonBackbone(), [665, 560, 490])
        self.assertEqual(dofa(image).shape, (2, 4, 2, 2))
        self.assertEqual(panopticon(image).shape, (2, 4, 2, 2))
        torch.testing.assert_close(dofa.backbone.patch_embed.last_waves, torch.tensor([0.665, 0.560, 0.490]))
        torch.testing.assert_close(
            panopticon.backbone.model.last_channel_ids,
            torch.tensor([[665.0, 560.0, 490.0], [665.0, 560.0, 490.0]]),
        )

    def test_common_decoder_produces_full_resolution_logits(self):
        decoder = CommonUNetDecoder(4, 3, decoder_channels=(8, 4))
        output = decoder(torch.randn(2, 4, 2, 2), output_size=(9, 11))
        self.assertEqual(output.shape, (2, 3, 9, 11))
        self.assertTrue(torch.isfinite(output).all())

    def test_common_decoder_mc_dropout_uses_only_final_feature_map_dropout2d(self):
        model = FoundationSegmentationModel(
            ToyDenseAdapter(),
            CommonUNetDecoder(4, 3, decoder_channels=(8, 4), dropout=0.1),
            freeze_backbone=True,
        )
        audit = audit_segmentation_model(
            model,
            {
                "adaptation_mode": "frozen",
                "decoder": {"architecture": "unet", "channels": [8, 4], "dropout": 0.1},
            },
        )
        self.assertEqual(audit["mc_dropout"][0]["class"], "Dropout2d")
        inference_audit = activate_downstream_mc_dropout(model)
        self.assertFalse(model.backbone.training)
        self.assertTrue(model.head.mc_dropout.training)
        self.assertEqual(inference_audit["batchnorm_training_modules"], [])
        self.assertEqual(inference_audit["backbone_stochastic_training_modules"], [])

    def test_frozen_and_full_finetune_gradient_contracts(self):
        for frozen in (True, False):
            model_cfg = {
                "adaptation_mode": "frozen" if frozen else "full_finetune",
                "expected_frozen_backbone_parameters": [],
            }
            model = FoundationSegmentationModel(
                ToyDenseAdapter(), CommonUNetDecoder(4, 2, decoder_channels=(8, 4)), freeze_backbone=frozen
            )
            audit = audit_segmentation_model(model, model_cfg)
            self.assertEqual(audit["backbone_requires_grad_all_false"], frozen)
            logits = model(torch.randn(2, 3, 8, 8))
            loss = segmentation_loss(logits, torch.randint(0, 2, (2, 8, 8)), ignore_index=255)
            loss.backward()
            backbone_gradients = [parameter.grad for parameter in model.backbone.parameters()]
            decoder_gradients = [parameter.grad for parameter in model.head.parameters()]
            if frozen:
                self.assertTrue(all(gradient is None for gradient in backbone_gradients))
            else:
                self.assertTrue(all(gradient is not None for gradient in backbone_gradients))
                self.assertTrue(all(torch.isfinite(gradient).all() for gradient in backbone_gradients))
            self.assertTrue(all(gradient is not None for gradient in decoder_gradients))

    def test_common_metrics_and_ignore_pixels(self):
        target = torch.tensor([[[0, 0, 1], [0, 1, 1], [2, 2, 255]]])
        logits = torch.full((1, 3, 3, 3), -4.0)
        for row in range(3):
            for column in range(3):
                label = int(target[0, row, column])
                if label != 255:
                    logits[0, label, row, column] = 4.0
        logits[:, :, 2, 2] = torch.tensor([100.0, -100.0, -100.0])
        metrics = segmentation_metrics(logits, target, ["a", "b", "c"], ignore_index=255, n_bins=15)
        self.assertEqual(metrics["miou"], 1.0)
        self.assertEqual(metrics["pixel_accuracy"], 1.0)
        self.assertEqual(metrics["ignored_pixels"], 1)
        self.assertEqual(metrics["valid_pixels"], 8)
        self.assertEqual(metrics["per_class_iou"], {"a": 1.0, "b": 1.0, "c": 1.0})
        self.assertLess(metrics["nll"], 0.001)
        self.assertLess(metrics["brier"], 0.001)
        self.assertTrue(math.isfinite(metrics["ece_15"]))

        changed = logits.clone()
        changed[:, :, 2, 2] = torch.tensor([-100.0, 100.0, -100.0])
        changed_metrics = segmentation_metrics(changed, target, ["a", "b", "c"], ignore_index=255)
        for name in ("miou", "pixel_accuracy", "nll", "brier", "ece_15"):
            self.assertAlmostEqual(metrics[name], changed_metrics[name], places=12)

    def test_flattened_segmentation_loss_matches_spatial_cross_entropy(self):
        logits = torch.randn(2, 3, 4, 5, requires_grad=True)
        target = torch.randint(0, 3, (2, 4, 5))
        target[0, 0, 0] = 255
        actual = segmentation_loss(logits, target, ignore_index=255)
        expected = torch.nn.functional.cross_entropy(logits, target, ignore_index=255)
        torch.testing.assert_close(actual, expected)

    def test_spacenet_foreground_classwise_and_boundary_calibration(self):
        target = torch.tensor([[[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [255, 0, 0, 0]]])
        logits = torch.zeros((1, 2, 4, 4))
        logits[:, 0] = 1.0
        logits[:, 1][target == 1] = 2.0
        metrics = segmentation_metrics(
            logits,
            target,
            ["background", "building"],
            ignore_index=255,
            n_bins=15,
            foreground_class_index=1,
            boundary_radius=1,
        )
        for name in ("foreground_ece_15", "foreground_nll", "foreground_brier"):
            self.assertTrue(math.isfinite(metrics[name]))
        self.assertEqual(set(metrics["classwise_calibration"]), {"background", "building"})
        self.assertGreater(metrics["boundary_calibration"]["pixel_count"], 0)
        self.assertTrue(math.isfinite(metrics["boundary_calibration"]["ece_15"]))
        self.assertTrue(math.isfinite(metrics["boundary_calibration"]["foreground_ece_15"]))

    def test_streaming_accumulator_matches_single_batch(self):
        logits = torch.randn(2, 2, 4, 4)
        target = torch.randint(0, 2, (2, 4, 4))
        target[0, 0, 0] = 255
        direct = segmentation_metrics(logits, target, ["background", "building"], 255, foreground_class_index=1)
        accumulator = SegmentationMetricAccumulator(2, ["background", "building"], 255, foreground_class_index=1)
        accumulator.update(logits[:1], target[:1])
        accumulator.update(logits[1:], target[1:])
        streamed = accumulator.compute()
        for name in (
            "miou",
            "pixel_accuracy",
            "nll",
            "brier",
            "ece_15",
            "foreground_ece_15",
            "foreground_nll",
            "foreground_brier",
        ):
            self.assertAlmostEqual(direct[name], streamed[name], places=6)

    def test_segmentation_prediction_export_roundtrip(self):
        logits = torch.randn(2, 2, 4, 4)
        target = torch.randint(0, 2, (2, 4, 4))
        target[0, 0, 0] = 255
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "prediction_export"
            paths = export_segmentation_predictions(
                output,
                sample_ids=["sample_a", "sample_b"],
                masks=target,
                logits=logits,
                class_names=["background", "building"],
                ignore_index=255,
                model_name="toy",
                dataset="spacenet7",
                adaptation_mode="frozen",
                split="val",
                checkpoint="toy.pt",
            )
            self.assertEqual(set(paths), {"bundle", "manifest", "validation"})
            validation = validate_segmentation_prediction_export(output)
            self.assertTrue(validation["valid"], validation)
            self.assertEqual(validation["sample_count"], 2)
            self.assertEqual(validation["ignored_pixels"], 1)

    def test_segmentation_prediction_export_with_representations_and_per_image_metrics(self):
        logits = torch.randn(2, 2, 4, 4)
        target = torch.randint(0, 2, (2, 4, 4))
        rows = [{"sample_id": "sample_a", "miou": 0.5}, {"sample_id": "sample_b", "miou": 0.75}]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "prediction_export"
            paths = export_segmentation_predictions(
                output,
                sample_ids=["sample_a", "sample_b"],
                masks=target,
                logits=logits,
                class_names=["background", "building"],
                ignore_index=255,
                model_name="toy",
                dataset="spacenet7",
                adaptation_mode="frozen",
                split="test",
                checkpoint="toy.pt",
                representations=torch.randn(2, 4),
                per_image_results=rows,
                source={"seed": 42},
            )
            self.assertEqual(
                set(paths),
                {"bundle", "representations", "per_image_metrics", "manifest", "validation"},
            )
            validation = validate_segmentation_prediction_export(output)
            self.assertTrue(validation["valid"], validation)
            self.assertEqual(validation["representation_shape"], [2, 4])

    def test_collect_segmentation_predictions_includes_pooled_dense_features(self):
        model = FoundationSegmentationModel(
            ToyDenseAdapter(), CommonUNetDecoder(4, 2, decoder_channels=(8, 4)), freeze_backbone=True
        )
        loader = [
            {
                "image": torch.randn(2, 3, 8, 8),
                "mask": torch.randint(0, 2, (2, 8, 8)),
                "sample_id": ["a", "b"],
            }
        ]
        collected = collect_segmentation_predictions(
            model,
            loader,
            torch.device("cpu"),
            ["background", "building"],
            ignore_index=255,
            foreground_class_index=1,
        )
        self.assertEqual(collected["logits"].shape, (2, 2, 8, 8))
        self.assertEqual(collected["representations"].shape, (2, 4))
        self.assertEqual([row["sample_id"] for row in collected["per_image_results"]], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
