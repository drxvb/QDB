//! Exercises the real persistent probe executable, including rejection paths.
use quickdb_api::{OpenOptions, Store};
use quickdb_native::NativeStore;
use std::path::Path;
use std::process::Command;

fn invoke(
    executable: &std::ffi::OsStr,
    path: &Path,
    count: &str,
    mode: &str,
) -> std::process::Output {
    Command::new(executable)
        .arg(path)
        .arg(count)
        .arg(mode)
        .output()
        .expect("execute actual point_probe")
}

#[test]
#[ignore = "requires built release example and QDB_POINT_PROBE executable path"]
fn persistent_probe_cli_guards_and_reuse() -> Result<(), Box<dyn std::error::Error>> {
    let executable = std::env::var_os("QDB_POINT_PROBE").ok_or("QDB_POINT_PROBE is required")?;
    let root = tempfile::tempdir()?;
    for (count, mode, message) in [
        ("0", "build", "expected positive count"),
        ("32", "invalid", "expected positive count"),
        ("32", "run", "run requires a completed build"),
        ("18446744073709551615", "build", "expected positive count"),
    ] {
        let path = root.path().join(format!("reject-{count}-{mode}"));
        let output = invoke(&executable, &path, count, mode);
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains(message));
        assert!(!path.exists(), "rejected input must not create a store");
    }
    let incomplete = root.path().join("incomplete");
    std::fs::create_dir(&incomplete)?;
    std::fs::write(incomplete.join("sentinel"), "preserve")?;
    let output = invoke(&executable, &incomplete, "32", "build");
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("unmarked nonempty store"));
    assert_eq!(
        std::fs::read_to_string(incomplete.join("sentinel"))?,
        "preserve"
    );
    assert_eq!(std::fs::read_dir(&incomplete)?.count(), 1);
    std::fs::write(incomplete.join("point-probe-count"), "31")?;
    let output = invoke(&executable, &incomplete, "32", "run");
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("stored count differs"));
    assert_eq!(
        std::fs::read_to_string(incomplete.join("point-probe-count"))?,
        "31"
    );
    assert_eq!(std::fs::read_dir(&incomplete)?.count(), 2);

    let path = root.path().join("valid");
    for (mode, existing) in [("build", false), ("run", true), ("build", true)] {
        let output = invoke(&executable, &path, "32", mode);
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8(output.stdout)?;
        assert!(stdout.contains(&format!("mode={mode} existing={existing}")));
        assert!(stdout.contains("phase=after_reopen operation=get_row samples=10000"));
        assert!(stdout.contains("rebuilds=0"));
        assert_eq!(stdout.contains("POINT loaded"), !existing);
        assert_eq!(
            std::fs::read_to_string(path.join("point-probe-count"))?,
            "32"
        );
        let store = NativeStore::open(
            &path,
            &OpenOptions {
                create: false,
                sync_each_write: false,
            },
        )?;
        assert_eq!(store.count_rows("docs")?, 32);
        assert_eq!(store.count_vectors("emb")?, 32);
        assert_eq!(store.count_edges()?, 160);
    }
    println!("POINT_CLI guards=6 no_rejected_writes=true lifecycle_runs=3 rows=32 vectors=32 edges=160 reuse=true");
    Ok(())
}
