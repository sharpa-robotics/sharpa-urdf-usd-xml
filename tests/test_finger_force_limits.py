"""Run with: python -m unittest discover -s tests -v (requires mujoco, numpy)."""
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FINGER_CLASSES = {"CMC_joint", "PCMC_joint", "MCP_joint", "PIP_joint", "DIP_joint"}


def description(path):
    root = ET.parse(path).getroot()
    limits = {
        joint.get("name"): float(joint.find("limit").get("effort"))
        for joint in ET.parse(path.with_suffix(".urdf")).getroot().findall("joint")
        if joint.find("limit") is not None
    }
    fingers = {
        joint.get("name"): limits[joint.get("name")]
        for joint in root.findall(".//body/joint")
        if joint.get("class") in FINGER_CLASSES
    }
    # Keep the pre-existing dual-hand resource-path issue out of this regression.
    meshdir = path.parent.parent if path.parent.name == "dual_sharpa_wave" else path.parent / root.find("compiler").get("meshdir", "")
    root.find("compiler").set("meshdir", str(meshdir))
    return root, fingers


def compile_model(root):
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def without_limits(root):
    root = ET.fromstring(ET.tostring(root))
    for default in root.findall(".//default"):
        if default.get("class") in FINGER_CLASSES:
            joint = default.find("joint")
            joint.attrib.pop("actuatorfrcrange", None)
    return compile_model(root)


class FingerForceLimitsTest(unittest.TestCase):
    def test_all_variants_match_urdf_and_clamp_both_directions(self):
        paths = [p for p in sorted(ROOT.rglob("*.xml")) if ET.parse(p).getroot().tag == "mujoco"]
        self.assertEqual(len(paths), 15)
        rng = np.random.default_rng(827624)
        for path in paths:
            with self.subTest(model=str(path.relative_to(ROOT))):
                root, fingers = description(path)
                model = compile_model(root)
                original = without_limits(root)
                self.assertEqual(len(fingers), 44 if path.parent.name == "dual_sharpa_wave" else 22)
                ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in fingers]
                expected = np.array(list(fingers.values()))
                self.assertTrue(np.all(model.jnt_actfrclimited[ids]))
                np.testing.assert_allclose(model.jnt_actfrcrange[ids], np.column_stack((-expected, expected)))
                # Gains, damping, inertia, geometry and virtual-base limits must stay unchanged.
                for attr in ("body_mass", "body_inertia", "jnt_range", "dof_armature", "dof_damping",
                             "dof_frictionloss", "actuator_gainprm", "actuator_biasprm",
                             "actuator_ctrlrange", "actuator_gear", "geom_contype", "geom_conaffinity"):
                    np.testing.assert_array_equal(getattr(model, attr), getattr(original, attr))
                base_ids = np.array([i for i in range(model.njnt) if i not in ids], dtype=int)
                np.testing.assert_array_equal(model.jnt_actfrclimited[base_ids], original.jnt_actfrclimited[base_ids])
                np.testing.assert_array_equal(model.jnt_actfrcrange[base_ids], original.jnt_actfrcrange[base_ids])
                dofs = model.jnt_dofadr[ids]
                data = mujoco.MjData(model)
                before = mujoco.MjData(original)
                for direction in (0, 1):
                    mujoco.mj_resetData(model, data)
                    data.ctrl[:] = model.actuator_ctrlrange[:, direction]
                    data.qvel[:] = -20 if direction else 20
                    mujoco.mj_forward(model, data)
                    self.assertTrue(np.all(np.abs(data.qfrc_actuator[dofs]) <= expected + 1e-10))
                    np.testing.assert_allclose(data.qfrc_actuator[dofs], expected * (1 if direction else -1))
                for _ in range(20):
                    mujoco.mj_resetData(model, data)
                    data.qpos[:] = rng.uniform(model.jnt_range[:, 0], model.jnt_range[:, 1])
                    data.qvel[:] = rng.uniform(-15, 15, model.nv)
                    data.ctrl[:] = rng.uniform(model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
                    before.qpos[:] = data.qpos
                    before.qvel[:] = data.qvel
                    before.ctrl[:] = data.ctrl
                    mujoco.mj_forward(model, data)
                    mujoco.mj_forward(original, before)
                    np.testing.assert_allclose(data.qfrc_actuator[dofs], np.clip(before.qfrc_actuator[dofs], -expected, expected))
                mujoco.mj_resetData(model, data)
                for _ in range(1000):
                    mujoco.mj_step(model, data)
                self.assertTrue(np.isfinite(data.qpos).all())
                self.assertFalse(data.warning.number.any())

    def test_sustained_load_and_unsaturated_motion(self):
        root, _ = description(ROOT / "wave_01/right_sharpa_wave/right_sharpa_wave.xml")
        fixed, original = compile_model(root), without_limits(root)
        jid = mujoco.mj_name2id(fixed, mujoco.mjtObj.mjOBJ_JOINT, "right_index_DIP")
        aid = mujoco.mj_name2id(fixed, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_index_DIP_ctrl")
        qid, dof = fixed.jnt_qposadr[jid], fixed.jnt_dofadr[jid]
        results = []
        # An external opposing joint torque isolates load-bearing ability from contact geometry.
        for target, load in ((0.05, 0), (0.7, -0.05), (0.7, -0.3)):
            pair = []
            for model in (original, fixed):
                model.opt.gravity[:] = 0
                model.geom_contype[:] = 0
                model.geom_conaffinity[:] = 0
                data = mujoco.MjData(model)
                data.ctrl[aid] = target
                data.qfrc_applied[dof] = load
                trajectory = []
                for _ in range(2000):
                    mujoco.mj_step(model, data)
                    trajectory.append(data.qpos[qid])
                self.assertFalse(data.warning.number.any())
                pair.append((data.qpos[qid], data.qfrc_actuator[dof], np.array(trajectory)))
            results.append(pair)
        np.testing.assert_allclose(results[0][0][2], results[0][1][2], atol=1e-12)
        self.assertAlmostEqual(results[1][0][0], results[1][1][0], places=5)
        self.assertGreater(results[2][0][0], 0.3)
        self.assertLess(results[2][1][0], 0.01)
        self.assertAlmostEqual(results[2][1][1], 0.189369, places=6)


if __name__ == "__main__":
    unittest.main()
