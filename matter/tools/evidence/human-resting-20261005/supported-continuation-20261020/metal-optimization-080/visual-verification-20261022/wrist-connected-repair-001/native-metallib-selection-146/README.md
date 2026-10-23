# Select the same Matter and Human shader library

The native app previously passed its build-time metallib to Human load adapters even when NUMI_MATTER_METALLIB_OVERRIDE selected another library for Matter. The mismatch can prevent new pipelines from encoding or execute a shader with an incompatible work domain. All adapter paths now share the selected library, with the environment override taking precedence over the optional CLI path exactly as in Matter. Startup records its path and SHA256.

The narrow patch is applied to main12569737 with every sibling header extracted from that commit; compile/link pass. The same selector block was executed in the142 all-bank integration run and logged the exact143 library hash; that run then failed a physical admission gate, with zero accepted steps. This patch fixes library selection and does not establish successful anatomy or mechanics qualification.

The first146 compile failed because a partial sibling-header copy caused mixed physical include identities; the full exact-parent app header closure compiles. Failed compile evidence is retained. No anatomy headers or contact laws are changed in this commit.
