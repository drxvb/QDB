# Test fixtures

`corpus-shapes.tsv` is **synthetic**: 22,666 rows of made-up payload shapes, written by a seeded generator. It holds no
data from any real corpus. The columns are Unicode character count (the indexed `size` value), UTF-8 body byte count,
UTF-8 title byte count, sequence number within a document, and corpus ordinal.

`corpus_range_query_plan_and_late_materialization` (in `src/range_repair_tests.rs`) builds the relational schema with
placeholder text of exactly these byte lengths, runs 500 range queries, and checks that the query plan uses the covering
index and that late materialization returns the same rows as the plain query.

`snap/*.bin` are small frozen snapshot files that pin the on-disk format across versions; the test that reads each one
says how it was produced.
