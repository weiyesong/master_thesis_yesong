import unittest
from unittest.mock import patch
from pathlib import Path

import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from scripts.run_experiments import (
    DOFARGBLinearProbe,
    PanopticonClassifier,
    audit_model_for_training,
    build_model,
    build_optimizer,
    classification_type_from_config,
    compute_task_classification_metrics,
    capture_dataloader_generator_states,
    capture_rng_state,
    configured_wavelengths,
    mean_temporal_encoder_features,
    metric_improved,
    reconstruct_classification_loader_states,
    restore_dataloader_generator_states,
    restore_rng_state,
    set_epoch_learning_rates,
    train_one_epoch,
)
from scripts.mc_dropout import activate_downstream_mc_dropout
from scripts.experiment_manager import resolve_config


class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 4)

    def forward_features(self, image, wave_list):
        del wave_list
        return self.projection(image.mean(dim=(-1, -2)))


class FakePanopticonBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 4)
        self.last_channel_ids = None

    def forward(self, inputs):
        self.last_channel_ids = inputs["chn_ids"].detach().clone()
        return self.projection(inputs["imgs"].mean(dim=(-1, -2)))


class FakeOfficialPanopticon(FakePanopticonBackbone):
    def __init__(self):
        super().__init__()
        self.model = type("ModelMetadata", (), {"num_features": 4})()


class ExperimentConfigurationTests(unittest.TestCase):
    def test_final_segmentation_configs_freeze_b5_protocol(self):
        root = Path(__file__).resolve().parents[1]
        cases = {
            "cloudsen12_frozen_final.yaml": ("frozen", 10),
            "cloudsen12_full_finetune_final.yaml": ("full_finetune", 15),
            "spacenet7_frozen_final.yaml": ("frozen", 10),
            "spacenet7_full_finetune_final.yaml": ("full_finetune", 15),
        }
        for filename, (adaptation, patience) in cases.items():
            path = root / "configs" / filename
            config = resolve_config(yaml.safe_load(path.read_text(encoding="utf-8")), path)
            self.assertEqual(config["seeds"], [42, 43, 44])
            self.assertEqual(config["training"]["batch_size"], 8)
            self.assertEqual(config["training"]["epochs"], 50)
            self.assertEqual(config["training"]["weight_decay"], 0.01)
            self.assertEqual(config["training"]["head_learning_rate"], 1e-3)
            self.assertEqual(config["training"]["backbone_learning_rate"], 1e-4)
            self.assertEqual(config["training"]["checkpoint"], {"split": "val", "metric": "miou", "mode": "max"})
            self.assertEqual(config["training"]["early_stopping"]["patience"], patience)
            self.assertTrue(config["prediction_export"]["enabled"])
            self.assertTrue(config["provenance"]["require_input_hashes"])
            self.assertTrue(all(item["model"]["adaptation_mode"] == adaptation for item in config["experiments"]))

    def test_rng_state_round_trip(self):
        torch.manual_seed(123)
        state = capture_rng_state()
        expected = torch.rand(4)
        restore_rng_state(state)
        self.assertTrue(torch.equal(torch.rand(4), expected))

    def test_loader_generator_state_reconstruction_matches_completed_epochs(self):
        def loaders():
            train_generator = torch.Generator().manual_seed(43)
            val_generator = torch.Generator().manual_seed(44)
            dataset = TensorDataset(torch.arange(17))
            return {
                "train": DataLoader(dataset, batch_size=4, shuffle=True, generator=train_generator),
                "val": DataLoader(dataset, batch_size=4, shuffle=False, generator=val_generator),
            }

        uninterrupted = loaders()
        for _ in range(5):
            list(uninterrupted["train"])
            list(uninterrupted["val"])
        expected = capture_dataloader_generator_states(uninterrupted)

        reconstructed = loaders()
        reconstruct_classification_loader_states(reconstructed, completed_epochs=5)
        actual = capture_dataloader_generator_states(reconstructed)
        self.assertTrue(torch.equal(actual["train"], expected["train"]))
        self.assertTrue(torch.equal(actual["val"], expected["val"]))

        reset = loaders()
        restore_dataloader_generator_states(reset, expected)
        self.assertTrue(torch.equal(reset["train"].generator.get_state(), expected["train"]))

    def test_shared_temporal_protocol_encodes_each_timestamp_then_means_features(self):
        image = torch.tensor(
            [[[[[1.0]], [[2.0]], [[3.0]]], [[[3.0]], [[4.0]], [[5.0]]]]]
        )

        def encoder(flattened):
            return flattened.flatten(2).mean(dim=2)

        features = mean_temporal_encoder_features(image, encoder)
        self.assertTrue(torch.equal(features, torch.tensor([[2.0, 3.0, 4.0]])))
        masked = mean_temporal_encoder_features(image, encoder, torch.tensor([[True, False]]))
        self.assertTrue(torch.equal(masked, torch.tensor([[1.0, 2.0, 3.0]])))

    def test_dofa_and_panopticon_share_temporal_and_head_contracts(self):
        image = torch.arange(2 * 3 * 3 * 2 * 2, dtype=torch.float32).reshape(2, 3, 3, 2, 2)
        wrappers = [
            DOFARGBLinearProbe(
                FakeBackbone(), 4, 5, [0.665, 0.560, 0.490], freeze_backbone=True,
                head_config={"architecture": "linear", "dropout": 0.0},
            ),
            PanopticonClassifier(
                FakePanopticonBackbone(), 4, 5, [665, 560, 490], freeze_backbone=True,
                head_config={"architecture": "linear", "dropout": 0.0},
            ),
        ]
        for model in wrappers:
            temporal = model.extract_features(image)
            expected = torch.stack(
                [model._encode_single_timestamp(image[:, timestamp]) for timestamp in range(image.shape[1])], dim=1
            ).mean(dim=1)
            self.assertTrue(torch.allclose(temporal, expected))
            self.assertEqual(model(image).shape, (2, 5))
            self.assertEqual([type(module) for module in model.head], [nn.Dropout, nn.Linear])

    def test_multilabel_loss_metrics_and_training_path(self):
        config = {"task": "classification_multilabel", "data": {"multilabel": True}}
        self.assertEqual(classification_type_from_config(config), "multilabel")
        logits = torch.tensor([[2.0, -2.0, 1.0], [-1.0, 2.0, -2.0]])
        labels = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        metrics = compute_task_classification_metrics(logits, labels, "multilabel", n_bins=15)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)

        model = DOFARGBLinearProbe(FakeBackbone(), 4, 3, [1.0, 2.0, 3.0], freeze_backbone=True)
        optimizer = build_optimizer(
            model,
            {
                "optimizer": {"name": "adamw"},
                "backbone_learning_rate": 1e-4,
                "head_learning_rate": 1e-3,
                "weight_decay": 0.0,
            },
        )
        result = train_one_epoch(
            model,
            [{"image": torch.ones(2, 3, 4, 4), "label": labels}],
            optimizer,
            torch.device("cpu"),
            classification_type="multilabel",
        )
        self.assertTrue(torch.isfinite(torch.tensor(result["loss"])))

    def test_panopticon_wrapper_uses_configured_channel_ids_and_adaptation(self):
        frozen_backbone = FakePanopticonBackbone()
        frozen = PanopticonClassifier(
            frozen_backbone,
            embed_dim=4,
            num_classes=2,
            channel_ids_nm=[665, 560, 490],
            freeze_backbone=True,
            head_config={"architecture": "linear", "dropout": 0.0},
        )
        self.assertEqual(frozen(torch.ones(2, 3, 4, 4)).shape, (2, 2))
        self.assertTrue(
            torch.equal(frozen_backbone.last_channel_ids, torch.tensor([[665.0, 560.0, 490.0]]).repeat(2, 1))
        )
        self.assertFalse(any(parameter.requires_grad for parameter in frozen.backbone.parameters()))

        trainable = PanopticonClassifier(
            FakePanopticonBackbone(),
            embed_dim=4,
            num_classes=2,
            channel_ids_nm=[665, 560, 490],
            freeze_backbone=False,
        )
        self.assertTrue(all(parameter.requires_grad for parameter in trainable.backbone.parameters()))

    def test_model_factory_selects_panopticon_without_downloading_in_test(self):
        config = {
            "data": {
                "image_size": 224,
                "input_bands": "rgb",
                "channels": ["B04", "B03", "B02"],
                "wavelengths_nm": [665, 560, 490],
            }
        }
        model_cfg = {
            "name": "panopticon",
            "size": "base",
            "pretrained": False,
            "adaptation_mode": "frozen",
            "freeze_backbone": True,
            "head": {"architecture": "linear", "dropout": 0.0},
        }
        with patch("torchgeo.models.panopticon_vitb14", return_value=FakeOfficialPanopticon()) as factory:
            model = build_model(config, model_cfg, num_classes=2)
        factory.assert_called_once_with(weights=None, img_size=224)
        self.assertIsInstance(model, PanopticonClassifier)
        self.assertEqual(model(torch.ones(2, 3, 4, 4)).shape, (2, 2))

    def test_panopticon_full_finetune_can_freeze_explicit_inactive_parameters(self):
        config = {
            "data": {
                "image_size": 224,
                "input_bands": "rgb",
                "channels": ["B04", "B03", "B02"],
                "wavelengths_nm": [664.63, 559.60, 493.00],
            }
        }
        model_cfg = {
            "name": "panopticon",
            "size": "base",
            "pretrained": False,
            "adaptation_mode": "full_finetune",
            "freeze_backbone": False,
            "expected_frozen_backbone_parameters": ["projection.weight"],
            "head": {"architecture": "linear", "dropout": 0.0},
        }
        with patch("torchgeo.models.panopticon_vitb14", return_value=FakeOfficialPanopticon()):
            model = build_model(config, model_cfg, num_classes=2)
        self.assertFalse(model.backbone.projection.weight.requires_grad)
        self.assertTrue(model.backbone.projection.bias.requires_grad)
        audit = audit_model_for_training(model, model_cfg)
        self.assertEqual(set(audit["backbone_structurally_frozen_parameters"]), {"projection.weight"})

    def test_wavelength_units_are_converted_per_model(self):
        config = {
            "data": {
                "input_bands": "rgb",
                "channels": ["B04", "B03", "B02"],
                "wavelengths_nm": [665, 560, 490],
            }
        }
        self.assertEqual(configured_wavelengths(config, "nanometers"), [665.0, 560.0, 490.0])
        self.assertEqual(configured_wavelengths(config, "micrometers"), [0.665, 0.56, 0.49])

    def test_wrong_canonical_wavelength_scale_is_rejected(self):
        config = {
            "data": {
                "input_bands": "rgb",
                "channels": ["B04", "B03", "B02"],
                "wavelengths_nm": [0.665, 0.560, 0.490],
            }
        }
        with self.assertRaisesRegex(ValueError, "nanometer scale"):
            configured_wavelengths(config, "micrometers")

    def test_frozen_backbone_and_configurable_mlp_head(self):
        model = DOFARGBLinearProbe(
            FakeBackbone(),
            embed_dim=4,
            num_classes=2,
            wavelengths=[1.0, 2.0, 3.0],
            freeze_backbone=True,
            head_config={"architecture": "mlp", "hidden_dim": 8, "dropout": 0.25},
        )
        self.assertFalse(any(parameter.requires_grad for parameter in model.backbone.parameters()))
        self.assertTrue(any(isinstance(module, nn.Dropout) and module.p == 0.25 for module in model.head.modules()))
        self.assertEqual(model(torch.ones(2, 3, 4, 4)).shape, (2, 2))

    def test_optimizer_uses_configured_head_learning_rate(self):
        model = DOFARGBLinearProbe(FakeBackbone(), 4, 2, [1.0, 2.0, 3.0], freeze_backbone=True)
        optimizer = build_optimizer(
            model,
            {
                "optimizer": {"name": "adamw"},
                "backbone_learning_rate": 1e-4,
                "head_learning_rate": 1e-3,
                "weight_decay": 1e-4,
            },
        )
        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertEqual(optimizer.param_groups[0]["lr"], 1e-3)

    def test_exact_frozen_batchnorm_linear_baseline(self):
        model_cfg = {
            "adaptation_mode": "frozen",
            "head": {
                "architecture": "batchnorm_linear",
                "batchnorm_affine": False,
                "batchnorm_eps": 1.0e-6,
                "dropout": 0.0,
            },
        }
        model = DOFARGBLinearProbe(
            FakeBackbone(),
            embed_dim=4,
            num_classes=2,
            wavelengths=[1.0, 2.0, 3.0],
            freeze_backbone=True,
            head_config=model_cfg["head"],
        )
        self.assertIsInstance(model.head[0], nn.BatchNorm1d)
        self.assertFalse(model.head[0].affine)
        self.assertEqual(model.head[0].eps, 1.0e-6)
        self.assertIsInstance(model.head[1], nn.Linear)
        self.assertFalse(any(isinstance(module, nn.Dropout) for module in model.head.modules()))
        audit = audit_model_for_training(model, model_cfg)
        self.assertEqual(audit["backbone_trainable_parameters"], 0)
        self.assertFalse(audit["training_mode_policy"]["backbone_training"])
        self.assertTrue(audit["training_mode_policy"]["head_training"])

    def test_batchnorm_linear_mc_dropout_is_the_only_active_inference_stochasticity(self):
        model_cfg = {
            "adaptation_mode": "frozen",
            "head": {
                "architecture": "batchnorm_linear",
                "batchnorm_affine": False,
                "batchnorm_eps": 1.0e-6,
                "dropout": 0.1,
            },
        }
        model = DOFARGBLinearProbe(
            FakeBackbone(), 4, 2, [1.0, 2.0, 3.0], freeze_backbone=True,
            head_config=model_cfg["head"],
        )
        self.assertEqual([type(module) for module in model.head], [nn.BatchNorm1d, nn.Dropout, nn.Linear])
        audit_model_for_training(model, model_cfg)
        inference_audit = activate_downstream_mc_dropout(model)
        self.assertFalse(model.head[0].training)
        self.assertTrue(model.head[1].training)
        self.assertFalse(model.backbone.training)
        self.assertEqual(inference_audit["batchnorm_training_modules"], [])
        self.assertEqual(inference_audit["backbone_stochastic_training_modules"], [])

    def test_full_finetune_parameter_groups_and_linear_warmup(self):
        model_cfg = {
            "adaptation_mode": "full_finetune",
            "head": {
                "architecture": "batchnorm_linear",
                "batchnorm_affine": False,
                "batchnorm_eps": 1.0e-6,
                "dropout": 0.0,
            },
        }
        model = DOFARGBLinearProbe(
            FakeBackbone(), 4, 2, [1.0, 2.0, 3.0], freeze_backbone=False, head_config=model_cfg["head"]
        )
        audit = audit_model_for_training(model, model_cfg)
        self.assertEqual(audit["backbone_trainable_parameters"], audit["backbone_total_parameters"])
        optimizer = build_optimizer(
            model,
            {
                "optimizer": {"name": "adamw"},
                "backbone_learning_rate": 4e-4,
                "head_learning_rate": 4e-3,
                "weight_decay": 0.0,
            },
        )
        self.assertEqual([group["name"] for group in optimizer.param_groups], ["backbone", "head"])
        epoch_one = set_epoch_learning_rates(optimizer, 1, {"enabled": True, "epochs": 5})
        self.assertAlmostEqual(epoch_one["backbone"], 8e-5)
        self.assertAlmostEqual(epoch_one["head"], 8e-4)
        epoch_five = set_epoch_learning_rates(optimizer, 5, {"enabled": True, "epochs": 5})
        self.assertAlmostEqual(epoch_five["backbone"], 4e-4)
        self.assertAlmostEqual(epoch_five["head"], 4e-3)

        result = train_one_epoch(
            model,
            [{"image": torch.ones(2, 3, 4, 4), "label": torch.tensor([0, 1])}],
            optimizer,
            torch.device("cpu"),
            audit_gradients=True,
        )
        self.assertTrue(result["gradient_audit"]["all_expected_parameters_received_gradient"])
        self.assertTrue(torch.isfinite(torch.tensor(result["backbone_gradient_norm"])))
        self.assertTrue(torch.isfinite(torch.tensor(result["head_gradient_norm"])))

    def test_metric_direction(self):
        self.assertTrue(metric_improved(0.2, 0.3, "min"))
        self.assertTrue(metric_improved(0.9, 0.8, "max"))
        self.assertFalse(metric_improved(0.3, 0.3, "min"))


if __name__ == "__main__":
    unittest.main()
