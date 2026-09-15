# Tuning status

There is no autotuner or schedule-analysis CLI in the current package.
Compilation retains the compiler's real schedule JSON and generated source map
for inspection. A static schedule is not a hardware timing measurement.

The current broadcast fixes the custom-broadcast ring at 256. Unsupported ring
sizes fail validation. Changing that field alone does not implement a valid
alternate layout: input distribution, collection, output shape, and transfers
must agree.

The [MOA audit](moa2026.md) records candidate dimensions from the official
baseline. None constitutes a generated, correctness-validated MOA candidate.
Any retained future result needs its baseline commit, compiler/package versions,
parameters, command, static schedule metric, and separate hardware correctness
and timing evidence.
