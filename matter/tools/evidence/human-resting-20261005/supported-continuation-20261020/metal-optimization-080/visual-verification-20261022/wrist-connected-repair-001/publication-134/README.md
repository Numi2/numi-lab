# Connected clearance: combined sparse solve
Tested485 combines exact sparse component decomposition with470 removal of artificial soft-target no-regression constraints. Physical hard rows, bounds, weighted target-slack cap and Float32 geometry gates remain required. Five helper tests pass. A component stage2 timeout uses a certified stage1 fallback, never reported as convergence.

Matched base461:3117 surface pairs and316 contained pairs. Checked487:3104 surface,298 contained,zero self pairs across185 banks.99 surface identities resolved,86 introduced;18 contained identities resolved,none introduced. Maximum depth2.164612 to1.978964mm. Independent proof preserves237 anchors,169 noncoordinate arrays,full146240 self-seed and13950 fixed-guard rows. Fit486153.227s; exact selection48760.329s. This is offline geometry construction,not simulation throughput.

Separate482 candidate3116/307 and its proof retained for comparison.479 remains3104/308,4463212/283. No mixed minima are claimed.
Zero crossing goal remains unmet.45 open head boundary edges,skin/fat cell validity,mass/thickness,conservative mechanical transfer,eight poses,and native16/60s qualification remain open. No visual verification or native step. No physical owner changed.
