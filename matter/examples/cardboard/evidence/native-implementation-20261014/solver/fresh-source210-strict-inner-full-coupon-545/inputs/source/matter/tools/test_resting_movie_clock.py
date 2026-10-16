"""Execute the movie inspector's actual Swift clock guard on macOS."""
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest


@unittest.skipUnless(sys.platform == "darwin" and shutil.which("swiftc"), "requires native Swift")
class RestingMovieClockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.binary = pathlib.Path(cls.temp.name) / "clock-check"
        source = pathlib.Path(__file__).with_name("inspect_resting_movie.swift").read_text()
        # Compile the inspector's exact clock function, without its AV reader.
        prefix = source.split("\nlet root =", 1)[0]
        driver = r"""
let input = try JSONSerialization.jsonObject(with:
    CommandLine.arguments[1].data(using: .utf8)!) as! [[Double]]
do {
    let clocks = try validateMovieClocks(input[0], input[1])
    let out = try JSONSerialization.data(withJSONObject:
        ["max_gap": clocks.maxGap, "startup_holds": clocks.startupHolds])
    print(String(data: out, encoding: .utf8)!)
} catch {
    print((error as NSError).domain)
    exit(3)
}
"""
        test = pathlib.Path(cls.temp.name) / "main.swift"
        test.write_text(prefix + driver)
        subprocess.run(["swiftc", str(test), "-o", str(cls.binary)],
                       check=True, capture_output=True, text=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def invoke(self, wall, simulated):
        return subprocess.run([str(self.binary), json.dumps([wall, simulated], allow_nan=False)],
                              capture_output=True, text=True)

    def test_initial_zero_frames_are_retained_and_counted(self):
        result = self.invoke([0, .1, .4, .9], [0, 0, .002, .004])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(json.loads(result.stdout), {"max_gap": .5, "startup_holds": 1})

    def test_startup_only_movie_and_one_frame(self):
        for wall, simulated, holds in (([0, .1], [0, 0], 1), ([0], [0], 0)):
            result = self.invoke(wall, simulated)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(json.loads(result.stdout)["startup_holds"], holds)

    def test_positive_clock_must_advance_and_never_return_to_zero(self):
        for simulated in ([0, .002, .002], [0, .004, .002], [0, .002, 0]):
            result = self.invoke([0, .1, .2], simulated)
            self.assertEqual(result.returncode, 3)
            self.assertIn("nonmonotonic-presentation", result.stdout)

    def test_wall_timestamps_must_still_increase(self):
        for wall in ([0, 0, .2], [0, .2, .1]):
            result = self.invoke(wall, [0, 0, .002])
            self.assertEqual(result.returncode, 3)
            self.assertIn("nonmonotonic-presentation", result.stdout)

    def test_empty_mismatched_and_negative_clocks_rejected(self):
        for wall, simulated in (([], []), ([0], []), ([-1], [0]), ([0], [-.002])):
            result = self.invoke(wall, simulated)
            self.assertEqual(result.returncode, 3)
            self.assertIn("invalid-presentation-clocks", result.stdout)


if __name__ == "__main__":
    unittest.main()
