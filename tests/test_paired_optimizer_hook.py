"""Exercise the trainer's actual nested guard through the installed optimizer API."""

import ast
from pathlib import Path
import unittest

import torch
from accelerate import Accelerator


ROOT = Path(__file__).resolve().parents[1]


def load_actual_guard(teacher, raw_student):
    """Compile the function node from the training file, so the test uses its real body/signature."""
    tree = ast.parse((ROOT / "scripts/train_paired_arm.py").read_text(encoding="utf-8"))
    guards = [node for node in ast.walk(tree)
              if isinstance(node, ast.FunctionDef) and node.name == "guard_gradient"]
    if len(guards) != 1:
        raise AssertionError("Expected exactly one trainer gradient guard")
    module = ast.fix_missing_locations(ast.Module(body=[guards[0]], type_ignores=[]))
    namespace = {"teacher": teacher, "raw_student": raw_student}
    exec(compile(module, str(ROOT / "scripts/train_paired_arm.py"), "exec"), namespace)
    return namespace["guard_gradient"]


class PairedOptimizerHookTests(unittest.TestCase):
    def test_real_accelerated_adamw_step_invokes_guard_and_preserves_rejections(self):
        student = torch.nn.Linear(3, 1)
        student.bias.requires_grad_(False)
        teacher = torch.nn.Linear(3, 1)
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)

        accelerator = Accelerator(cpu=True, mixed_precision="no")
        base_optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=0.01)
        student, optimizer = accelerator.prepare(student, base_optimizer)
        raw_student = accelerator.unwrap_model(student)
        guard = load_actual_guard(teacher, raw_student)
        optimizer.optimizer.register_step_pre_hook(guard)

        optimizer.zero_grad()
        loss = student(torch.ones(4, 3)).square().mean()
        accelerator.backward(loss)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in raw_student.parameters() if p.requires_grad))
        before = raw_student.weight.detach().clone()
        optimizer.step()
        self.assertFalse(torch.equal(before, raw_student.weight.detach()))
        self.assertEqual(int(optimizer.optimizer.state[raw_student.weight]["step"].item()), 1)

        optimizer.zero_grad()
        accelerator.backward(student(torch.ones(4, 3)).square().mean())
        teacher.weight.grad = torch.ones_like(teacher.weight)
        with self.assertRaisesRegex(RuntimeError, "Teacher received gradient"):
            optimizer.step()
        self.assertEqual(int(optimizer.optimizer.state[raw_student.weight]["step"].item()), 1)

        teacher.weight.grad = None
        raw_student.bias.grad = torch.ones_like(raw_student.bias)
        with self.assertRaisesRegex(RuntimeError, "Frozen student parameter received gradient"):
            optimizer.step()
        self.assertEqual(int(optimizer.optimizer.state[raw_student.weight]["step"].item()), 1)


if __name__ == "__main__":
    unittest.main()