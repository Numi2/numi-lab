"""CPU specification oracle; does not invoke or modify the Metal owner."""
from dataclasses import dataclass, replace
import unittest

@dataclass(frozen=True)
class State:
    count: int = 0
    held: int | None = None
    history: tuple = ()
    episode: int = 0

def candidate(accepted, new_frame, reset=False, frame_stride=2):
    # Frame rate is expressed in accepted environment steps, not submissions.
    if reset:
        accepted = State(episode=accepted.episode + 1)
    held = new_frame if accepted.count % frame_stride == 0 else accepted.held
    return replace(accepted, count=accepted.count + 1, held=held,
                   history=(accepted.history + (held,))[-3:])

def publish(accepted, proposed, success, command_succeeded=True):
    return [p if command_succeeded and ok else old
            for old, p, ok in zip(accepted, proposed, success, strict=True)]

class TransactionOracle(unittest.TestCase):
    def test_mixed_failure_retry_and_cadence(self):
        states = [State(), State()]
        # Each pair is the next real camera frame offered to that environment.
        states = publish(states, [candidate(s, f) for s, f in zip(states, [10, 20])], [True, True])
        states = publish(states, [candidate(s, f) for s, f in zip(states, [11, 21])], [True, True])
        self.assertEqual(states, [State(2,10,(10,10)), State(2,20,(20,20))])
        prior_failed_env = states[0]
        trial = [candidate(s, f) for s, f in zip(states, [12,22])]
        self.assertEqual(trial[0], State(3,12,(10,10,12)))
        states = publish(states, trial, [False,True])
        self.assertEqual(states, [prior_failed_env, State(3,22,(20,20,22))])
        # A retry of env 0 uses the same accepted count and initial frame;
        # env 1 advances independently. The scalar batch cadence cannot do this.
        states = publish(states, [candidate(states[0],12), candidate(states[1],23)], [True,True])
        reference = State(3,12,(10,10,12))
        self.assertEqual(states[0],reference)
        self.assertEqual(states[1],State(4,22,(20,22,22)))

    def test_different_environment_sample_phases(self):
        # One environment was rejected twice while another advanced.
        # Identical wall submissions must not force identical sample phases.
        accepted=[State(2, 10, (10,10)), State(4, 20, (20,20,20))]
        trial=[candidate(accepted[0],12), candidate(accepted[1],24)]
        accepted=publish(accepted,trial,[False,True])
        self.assertEqual(accepted,[State(2,10,(10,10)), State(5,24,(20,20,24))])
        next_trial=[candidate(accepted[0],12), candidate(accepted[1],25)]
        self.assertEqual(next_trial[0].held,12)
        self.assertEqual(next_trial[1].held,24)
        self.assertEqual(publish(accepted,next_trial,[True,True]),
                         [State(3,12,(10,10,12)), State(6,24,(20,24,24))])

    def test_rejected_reset_and_command_failure_are_atomic(self):
        old=[State(5,30,(20,20,30),2),State(2,50,(50,50),7)]
        trial=[candidate(old[0],40,reset=True),candidate(old[1],60)]
        self.assertEqual(publish(old,trial,[False,True]),
                         [old[0],State(3,60,(50,50,60),7)])
        self.assertEqual(publish(old,trial,[True,True],False),old)
        self.assertEqual(publish(old,trial,[True,False])[0],State(1,40,(40,),3))

if __name__ == "__main__":
    unittest.main(verbosity=2)
