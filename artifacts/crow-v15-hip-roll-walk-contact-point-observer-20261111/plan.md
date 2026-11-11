# Gait contact-point observer

Add world-space contact points to accepted-endpoint ground contact diagnostics, preserving the hip-roll v15 body, solver and substep observer. The observer must reproduce q/v and existing contact values exactly with point fields excluded before any slip measurement can be interpreted. Contact-point velocity is an accepted-endpoint estimate, not a continuous sliding bound.
