# Security policy

## Status

QDB 0.1.0 is pre-release software. It has had no external security review and no fuzzing. Do not use it to store data you
cannot afford to lose, or to open store directories you did not create. Opening a damaged or hostile store directory is a
supported failure case that should return an error, but that is not yet established by anything stronger than the tests in
`crates/quickdb-native/tests/`.

## Reporting a vulnerability

Please report privately and do not open a public issue: use GitHub's "Report a vulnerability" (private security advisory)
on this repository. Include the version or commit, a description, and the smallest input or store directory that reproduces
the problem.

## Supported versions

Only the latest commit on the main branch. There are no released versions yet.

## Scope

In scope: memory safety, data corruption or loss that the documented durability rules say should not happen, panics on
malformed input through the `Store` API, and unsafe handling of paths (a store must write only inside its directory).

Out of scope: weaknesses in SQLite itself (report them to the SQLite project), denial of service by supplying an input that
is large enough to need a lot of memory or time, and the comparison adapters (`quickdb-cozo`, `quickdb-lance`,
`quickdb-surreal`, `quickdb-sqlitevec`, `quickdb-sqlitevec-ann`, `quickdb-vec1`), which exist only to measure other engines.
