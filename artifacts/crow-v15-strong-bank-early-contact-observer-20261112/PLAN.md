# Early contact observer for failed flight trial

The retained first episode contains ground contact at intervals 1215 and 1258, well before termination at 1302. The previous observer began too late and cannot identify those contact body pairs. Repeat the same stronger-bank six-lane development experiment and passively log lane 4 body pairs, position-drive targets and substep joint state across 1200–1305, and dense control-interval state and policy actions from 1199–1305.

Require exact original first-episode sampled q/v/action and all per-substep contact summaries plus identical episode outcomes before treating the replay as matched. Identify the first recorded positive-impulse body pair and the preceding state, without conflating body IDs with shape IDs or declaring that contact caused the later failure. Position-drive targets are not complete wing forces, and raw policy actions are not necessarily applied fixed-controller actions.
