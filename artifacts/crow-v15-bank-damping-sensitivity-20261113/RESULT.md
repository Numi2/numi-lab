# Failed no-op damping sensitivity trial

The registered physical screen passed and the opposite signed-heading gate failed, exactly as in the prior bank experiment. A subsequent full-first-episode comparison found bit-identical recorded q, v, policy actions and contacts in all eight lanes against the original same-seed bank run. The intended damping intervention therefore cannot be treated as tested.

The source used pass.controlStep >= 550, but the host initializes that counter inside each dispatch and this rollout used --chunk 1, so the counter is local to each one-step submission. That explains why the intervention was not reached; the actual GPU parameter was not independently recorded in this run. The corrected episode-step intervention and GPU body-parameter readback were separately preregistered. This failure is retained and is not evidence that damping reduction has no effect.
