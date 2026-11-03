# Completed-Newton coupled nodal preconditioner retention

Selector6 stores its nodal inverse separately from the tail-cleared vector scratch. Reuse it only on intermediate iterations after the existing complete-Newton GPU latch freezes the candidate. The first iteration, final refresh and diagnostic captures keep ordinary work. Other selectors are unchanged. No equations, loads, constitutive arithmetic or acceptance gates change.

The native regression covers serial/cooperative builders, stationary completed tails, accepted motion, late rejection rollback and replay. Accepted state and certificates match bitwise.

The one matched full-Human four-step pair uses diagnostic Krylov16, not the64-budget default. Mean GPU time17.3342→13.5625seconds (21.76% lower time,1.2781x throughput). Inclusive elapsed257.40→233.19seconds. All833,232,924bytes of the existing endpoint capture match exactly (SHA25666c476e14b867eb9a19039ec112750a3d517a18ce039b24bdb5181fec7ac24c7); physiology CSV is also byte-identical. Capture record totals are declared native header totals, not an independent record counter. Full captures remain local; prefixes do not establish full-field parity by themselves.

The existing endpoint analysis initially refused native version2; this failed attempt is retained. Byte parity is independent of that parser. Later audit-source fixes are a separate increment.

This is an unbalanced release over62.5microseconds. GPU-only RTF1.15207e-6; inclusive RTF2.68016e-7. It does not establish sustained motion, settled rest, complete physiological/force qualification or0.1x. One pair is exploratory, not a three-pair performance qualification.
