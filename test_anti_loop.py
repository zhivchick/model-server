import unittest
import json
import re
from anti_loop import anti_loop_engine, AntiLoopEngine
from goose_hooks import apply_pre_call_hooks

class TestAntiLoopSuite(unittest.TestCase):
    def setUp(self):
        anti_loop_engine.hit_count = 0
        anti_loop_engine.last_tool = None
        anti_loop_engine.last_skeleton = None

    def test_github_issue_ids_exemption(self):
        engine = AntiLoopEngine()
        skel24 = engine._build_argument_skeleton({"command": "gh issue view 24"})
        skel25 = engine._build_argument_skeleton({"command": "gh issue view 25"})
        self.assertNotEqual(skel24, skel25, "GitHub issue numbers must produce different skeletons!")

        t1, _ = engine.evaluate_and_process("", "shell", {"command": "gh issue view 24"})
        self.assertEqual(t1, "shell")
        self.assertEqual(engine.hit_count, 0)

        t2, _ = engine.evaluate_and_process("", "shell", {"command": "gh issue view 25"})
        self.assertEqual(t2, "shell")
        self.assertEqual(engine.hit_count, 0, "Inspecting a different issue must not increase hit_count!")

    def test_github_pr_and_run_exemption(self):
        engine = AntiLoopEngine()
        skel_pr1 = engine._build_argument_skeleton({"command": "gh pr diff 105"})
        skel_pr2 = engine._build_argument_skeleton({"command": "gh pr diff 106"})
        self.assertNotEqual(skel_pr1, skel_pr2, "GitHub PR numbers must produce different skeletons!")

        skel_run1 = engine._build_argument_skeleton({"command": "gh run view 98765"})
        skel_run2 = engine._build_argument_skeleton({"command": "gh run view 98766"})
        self.assertNotEqual(skel_run1, skel_run2, "GitHub run numbers must produce different skeletons!")

        skel_api1 = engine._build_argument_skeleton({"command": "gh api repos/org/repo/issues/12"})
        skel_api2 = engine._build_argument_skeleton({"command": "gh api repos/org/repo/issues/13"})
        self.assertNotEqual(skel_api1, skel_api2, "GitHub API issue URLs must produce different skeletons!")

    def test_sliding_window_loops_still_detected(self):
        engine = AntiLoopEngine()
        skel_head1 = engine._build_argument_skeleton({"command": "head -n 10 file.txt"})
        skel_head2 = engine._build_argument_skeleton({"command": "head -n 20 file.txt"})
        self.assertEqual(skel_head1, skel_head2, "Sliding head reads must be caught as identical skeletons!")

        skel_sed1 = engine._build_argument_skeleton({"command": "sed -n '1,10p' file.txt"})
        skel_sed2 = engine._build_argument_skeleton({"command": "sed -n '11,20p' file.txt"})
        self.assertEqual(skel_sed1, skel_sed2, "Sliding sed reads must be caught as identical skeletons!")

        skel_list1 = engine._build_argument_skeleton({"command": "gh issue list -L 10"})
        skel_list2 = engine._build_argument_skeleton({"command": "gh issue list -L 20"})
        self.assertEqual(skel_list1, skel_list2, "Pagination limit changes must be caught as loops!")

    def test_three_tier_anti_loop_progression(self):
        engine = AntiLoopEngine()
        tool = "shell"
        args = {"command": "cat /tmp/nonexistent.log"}

        # Call 1: normal execution
        t1, _ = engine.evaluate_and_process("", tool, args)
        self.assertEqual(t1, "shell")
        self.assertEqual(engine.hit_count, 0)

        # Call 2: hit 1 -> Tier 1 (Tool error deflection)
        t2, a2 = engine.evaluate_and_process("", tool, args)
        self.assertEqual(t2, "shell")
        self.assertEqual(engine.hit_count, 1)
        self.assertIn("Execution Error", a2)

        # Call 3: hit 2 -> Tier 1 (Tool error deflection 2)
        t3, a3 = engine.evaluate_and_process("", tool, args)
        self.assertEqual(t3, "shell")
        self.assertEqual(engine.hit_count, 2)
        self.assertIn("Execution Error", a3)

        # Call 4: hit 3 -> Tier 2 (User Intervention level 1)
        t4, a4 = engine.evaluate_and_process("", tool, args)
        self.assertEqual(t4, "shell")
        self.assertEqual(engine.hit_count, 3)
        self.assertIn("[USER INTERVENTION]", a4)

        # Call 5: hit 4 -> Tier 2 (User Directive final warning)
        t5, a5 = engine.evaluate_and_process("", tool, args)
        self.assertEqual(t5, "shell")
        self.assertEqual(engine.hit_count, 4)
        self.assertIn("[USER DIRECTIVE - FINAL WARNING]", a5)

        # Call 6: hit 5+ -> Tier 3 (Hard Dialogue Brake)
        t6, a6 = engine.evaluate_and_process("", tool, args)
        self.assertIsNone(t6, "Hard threshold must return None tool_name to brake the dialogue!")
        self.assertIn("Critical repetition loop detected", a6)
        self.assertEqual(engine.hit_count, 0, "Hit count must reset after dialogue brake!")

    def test_pivot_resets_counter(self):
        engine = AntiLoopEngine()
        tool = "shell"
        args = {"command": "cat a.txt"}

        engine.evaluate_and_process("", tool, args)
        engine.evaluate_and_process("", tool, args) # hit 1
        self.assertEqual(engine.hit_count, 1)

        # Model pivots to a different file
        t_pivot, _ = engine.evaluate_and_process("", tool, {"command": "cat b.txt"})
        self.assertEqual(t_pivot, "shell")
        self.assertEqual(engine.hit_count, 0, "Pivoting to a new command must reset hit_count!")

    def test_goose_hooks_user_injection(self):
        anti_loop_engine.hit_count = 2
        anti_loop_engine.last_tool = "developer__edit"

        body = {
            "messages": [
                {"role": "user", "content": "Fix bug in app"},
                {"role": "assistant", "content": "I will edit the file"},
                {"role": "tool", "content": "Error: multiple matches"}
            ]
        }

        fixed_messages, kwargs = apply_pre_call_hooks(body)
        self.assertEqual(fixed_messages[-1]["role"], "user")
        self.assertIn("[USER INTERVENTION]", fixed_messages[-1]["content"])
        self.assertIn("developer__edit", fixed_messages[-1]["content"])

if __name__ == "__main__":
    unittest.main()
