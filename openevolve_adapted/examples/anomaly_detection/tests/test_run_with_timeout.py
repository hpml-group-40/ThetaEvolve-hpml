import signal
import subprocess
import unittest
from unittest.mock import patch

from examples.anomaly_detection import run_with_timeout


class RunWithTimeoutTests(unittest.TestCase):
    @patch("examples.anomaly_detection.run_with_timeout.subprocess.Popen")
    def test_returns_child_exit_code_before_deadline(self, popen):
        process = popen.return_value
        process.wait.return_value = 0

        result = run_with_timeout.run_with_timeout(["command"], 10)

        self.assertEqual(result, 0)
        process.terminate.assert_not_called()

    @patch("examples.anomaly_detection.run_with_timeout.os.killpg")
    @patch("examples.anomaly_detection.run_with_timeout.subprocess.Popen")
    def test_timeout_requests_shutdown_then_kills_stuck_process_group(self, popen, killpg):
        process = popen.return_value
        process.pid = 1234
        process.wait.side_effect = [
            subprocess.TimeoutExpired("command", 1),
            subprocess.TimeoutExpired("command", 20),
            -signal.SIGKILL,
        ]

        result = run_with_timeout.run_with_timeout(["command"], 1)

        self.assertEqual(result, 124)
        process.terminate.assert_called_once()
        killpg.assert_called_once_with(1234, signal.SIGKILL)


if __name__ == "__main__":
    unittest.main()