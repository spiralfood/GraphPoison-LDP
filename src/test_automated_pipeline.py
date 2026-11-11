import os
import tempfile
import unittest
from unittest import mock

import automated_pipeline


class AutomatedPipelineTests(unittest.TestCase):
    def test_create_results_backup_nests_under_dataset_tag(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_dirs = [
                "reconstructed_graph",
                "hmg_average_degrees",
                "output_py",
                "target_nodes_record",
                "HMG_attack",
                "attack_gain_results",
                "original_vs_attack_gain_results",
                "FakeNodeID",
                "HMG_FakeNode_Merged",
            ]

            for dir_name in sample_dirs:
                os.makedirs(os.path.join(tmpdir, dir_name), exist_ok=True)
                with open(os.path.join(tmpdir, dir_name, "marker.txt"), "w", encoding="utf-8") as handle:
                    handle.write(dir_name)

            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                automated_pipeline.create_results_backup(
                    beta=0.05,
                    target_ratio=0.1,
                    eps=4.0,
                    allocation_strategy=0,
                    dataset_tag="Synthetic",
                )
            finally:
                os.chdir(original_cwd)

            backup_root = os.path.join(tmpdir, "results_backup", "Synthetic")
            self.assertTrue(os.path.isdir(backup_root))

            strategy_name = automated_pipeline.get_strategy_name(0)
            safe_label = (
                strategy_name.replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
                .replace("theta=0.5", "theta_0_5")
            )
            folder_name = f"{safe_label}_strategy_0_beta_0.05_target_0.1_eps_4.0"
            backup_dir = os.path.join(backup_root, folder_name)

            self.assertTrue(os.path.isdir(backup_dir))
            self.assertTrue(os.path.isfile(os.path.join(backup_dir, "reconstructed_graph", "marker.txt")))


if __name__ == "__main__":
    unittest.main()
