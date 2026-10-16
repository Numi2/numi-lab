# Failed first analysis attempt

The first analyzer stopped before producing a report because it assumed the runtime emits CGLS summaries only after primary selection starts. The raw runtime emits64 summaries and selects the primary policy at48..63. The unchanged native attempt is analysed in002. This checker failure does not indicate a native replay or physics change.
