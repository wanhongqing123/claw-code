//! End-to-end checks that the packaged `claw` binary carries server-side content
//! blocks it does not model, and refuses to present a paused turn as a finished
//! answer.
//!
//! These drive the real CLI against a local mock service and read the captured
//! HTTP requests, because the defect they guard lives in the CLI's own stream
//! consumption and history building - not in the `claw-analog` helper path.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use mock_anthropic_service::{MockAnthropicService, SCENARIO_PREFIX};
use serde_json::Value;

struct Workspace {
    root: PathBuf,
}

impl Workspace {
    fn create(label: &str) -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after the epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("claw-passthrough-{label}-{nanos}"));
        fs::create_dir_all(root.join("config")).expect("config home should be created");
        fs::create_dir_all(root.join("home")).expect("home should be created");
        fs::write(root.join("fixture.txt"), "fixture body\n").expect("fixture should be written");
        Self { root }
    }

    fn config_home(&self) -> PathBuf {
        self.root.join("config")
    }

    fn home(&self) -> PathBuf {
        self.root.join("home")
    }
}

impl Drop for Workspace {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn run_claw(
    workspace: &Workspace,
    base_url: &str,
    scenario: &str,
    allowed_tools: Option<&str>,
) -> std::process::Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_claw"));
    command.current_dir(&workspace.root).env_clear();
    // Windows sockets fail to initialize without SystemRoot, so a fully cleared
    // environment cannot reach the mock server at all. Restore the minimum the
    // platform needs, and nothing that would leak real credentials.
    #[cfg(windows)]
    {
        for key in ["SystemRoot", "WINDIR", "SystemDrive", "PATH"] {
            if let Ok(value) = std::env::var(key) {
                command.env(key, value);
            }
        }
    }
    #[cfg(not(windows))]
    command.env("PATH", "/usr/bin:/bin");
    command
        .env("ANTHROPIC_API_KEY", "test-passthrough-key")
        .env("ANTHROPIC_BASE_URL", base_url)
        .env("CLAW_CONFIG_HOME", workspace.config_home())
        .env("HOME", workspace.home())
        .env("NO_COLOR", "1")
        .args([
            "--model",
            "sonnet",
            "--permission-mode",
            "read-only",
            "--output-format=json",
        ]);
    if let Some(tools) = allowed_tools {
        command.args(["--allowedTools", tools]);
    }
    command.arg(format!("{SCENARIO_PREFIX}{scenario}"));
    command.output().expect("claw should launch")
}

/// Parsed `/v1/messages` request bodies, oldest first.
fn message_request_bodies(server: &MockAnthropicService, runtime: &tokio::runtime::Runtime) -> Vec<Value> {
    runtime
        .block_on(server.captured_requests())
        .into_iter()
        .filter(|request| request.path.contains("/v1/messages") && !request.path.contains("count_tokens"))
        .map(|request| {
            serde_json::from_str::<Value>(&request.raw_body).expect("request body should be JSON")
        })
        .collect()
}

fn assistant_blocks(body: &Value) -> Vec<&Value> {
    body["messages"]
        .as_array()
        .expect("messages array")
        .iter()
        .filter(|message| message["role"] == "assistant")
        .filter_map(|message| message["content"].as_array())
        .flatten()
        .collect()
}

/// The regression this file exists for: a `server_tool_use` block (and its
/// result) must come back on the *next* request. Dropping them silently breaks
/// any turn the server expects to resume, and it is the CLI - not the analog
/// helper - that ships in the installer.
#[test]
fn cli_replays_server_tool_blocks_on_the_next_request() {
    let workspace = Workspace::create("replay");
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime should build");
    let server = runtime
        .block_on(MockAnthropicService::spawn())
        .expect("mock service should start");
    let base_url = server.base_url();

    let output = run_claw(
        &workspace,
        &base_url,
        "server_tool_passthrough",
        Some("read_file"),
    );
    assert!(
        output.status.success(),
        "claw should finish: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let bodies = message_request_bodies(&server, &runtime);
    assert!(
        bodies.len() >= 2,
        "the client tool call should force a second request, got {} ({bodies:#?})",
        bodies.len()
    );

    let replayed = assistant_blocks(&bodies[1]);
    let server_use = replayed
        .iter()
        .find(|block| block["type"] == "server_tool_use")
        .unwrap_or_else(|| {
            panic!("the server tool block must be replayed, got {replayed:#?}");
        });
    assert_eq!(server_use["id"], "srvtoolu_web");
    assert_eq!(server_use["name"], "webReader");
    assert_eq!(
        server_use["input"]["url"], "https://example.com",
        "the streamed input must survive, not the empty object from the start event"
    );
    assert!(
        replayed
            .iter()
            .any(|block| block["type"] == "web_search_tool_result"),
        "the server tool's result block must be replayed too, got {replayed:#?}"
    );

    // The ordinary content around it must be unharmed, and the client tool must
    // have received its own arguments rather than the server block's chunks.
    let client_use = replayed
        .iter()
        .find(|block| block["type"] == "tool_use")
        .expect("the client tool call must be replayed");
    assert_eq!(client_use["name"], "read_file");
    assert_eq!(
        client_use["input"]["path"], "fixture.txt",
        "each block's streamed input must stay with that block"
    );
    assert!(
        replayed.iter().any(|block| block["type"] == "text"
            && block["text"].as_str().unwrap_or_default().contains("checking the page")),
        "the assistant text must be preserved, got {replayed:#?}"
    );

    // And the server block must never be dispatched as a local tool.
    let response: Value =
        serde_json::from_slice(&output.stdout).expect("claw should emit JSON output");
    let dispatched: Vec<&str> = response["tool_uses"]
        .as_array()
        .map(|uses| {
            uses.iter()
                .filter_map(|use_| use_["name"].as_str())
                .collect()
        })
        .unwrap_or_default();
    assert_eq!(
        dispatched,
        vec!["read_file"],
        "only the client tool may be executed locally"
    );
}

/// Saving is only half the job: `ContentBlock::from_json` used to reject any
/// unmodeled block type, so a session that had merely seen a `server_tool_use`
/// became unloadable and every `--resume` failed outright.
///
/// Scope, stated because it is narrower than it looks: no `--resume` slash
/// command in this build issues an HTTP request (`/compact` summarizes locally;
/// `/summary` and `/retry` are unimplemented), and `--resume` refuses a plain
/// prompt. So this proves the block is persisted, that a restart loads it, and
/// that the reloaded session still holds it intact - the request-side replay is
/// covered by `cli_replays_server_tool_blocks_on_the_next_request`.
#[test]
fn cli_restart_reloads_a_session_containing_server_tool_blocks() {
    let workspace = Workspace::create("resume");
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime should build");
    let server = runtime
        .block_on(MockAnthropicService::spawn())
        .expect("mock service should start");
    let base_url = server.base_url();

    let first = run_claw(
        &workspace,
        &base_url,
        "server_tool_passthrough",
        Some("read_file"),
    );
    assert!(
        first.status.success(),
        "the first run should finish: {}",
        String::from_utf8_lossy(&first.stderr)
    );

    // The session file must have recorded the block, not dropped it. Its
    // directory is fingerprinted from the workspace, so search for it.
    let session_path = find_session_file(&workspace.root)
        .or_else(|| find_session_file(&workspace.config_home()))
        .or_else(|| find_session_file(&workspace.home()))
        .expect("the run should have written a session");
    let saved = fs::read_to_string(&session_path).expect("session should be readable");
    assert!(
        saved.contains("server_tool_use"),
        "the saved session must keep the server block: {saved}"
    );
    let session_id = session_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .expect("session id")
        .to_string();

    // A restart must load the session instead of failing on the unmodeled block.
    // `/status` is used deliberately: it exercises `Session::load` without any
    // model call, which is the step that used to fail.
    let mut resume = Command::new(env!("CARGO_BIN_EXE_claw"));
    resume.current_dir(&workspace.root).env_clear();
    #[cfg(windows)]
    {
        for key in ["SystemRoot", "WINDIR", "SystemDrive", "PATH"] {
            if let Ok(value) = std::env::var(key) {
                resume.env(key, value);
            }
        }
    }
    #[cfg(not(windows))]
    resume.env("PATH", "/usr/bin:/bin");
    let resumed = resume
        .env("ANTHROPIC_API_KEY", "test-passthrough-key")
        .env("ANTHROPIC_BASE_URL", base_url)
        .env("CLAW_CONFIG_HOME", workspace.config_home())
        .env("HOME", workspace.home())
        .env("NO_COLOR", "1")
        .args([
            "--model",
            "sonnet",
            "--permission-mode",
            "read-only",
            "--resume",
            &session_id,
        ])
        .arg("/status")
        .output()
        .expect("claw should launch");

    let rendered = format!(
        "{}{}",
        String::from_utf8_lossy(&resumed.stdout),
        String::from_utf8_lossy(&resumed.stderr)
    );
    assert!(
        resumed.status.success(),
        "resuming a session with a server block must not fail: {rendered}"
    );
    assert!(
        !rendered.contains("unsupported block type"),
        "the block must load rather than be rejected: {rendered}"
    );

    // And the reloaded session must still hold the block, with its input intact -
    // loading without error would otherwise be satisfied by silently dropping it.
    let reloaded = runtime::Session::load_from_path(&session_path)
        .expect("the saved session must load through the same path the CLI uses");
    let passthroughs: Vec<&String> = reloaded
        .messages
        .iter()
        .flat_map(|message| message.blocks.iter())
        .filter_map(|block| match block {
            runtime::ContentBlock::Passthrough { json } => Some(json),
            _ => None,
        })
        .collect();
    assert_eq!(
        passthroughs.len(),
        2,
        "both server blocks must survive the round trip: {:#?}",
        reloaded.messages
    );
    let server_use: Value =
        serde_json::from_str(passthroughs[0]).expect("reloaded block should be JSON");
    assert_eq!(server_use["type"], "server_tool_use");
    assert_eq!(
        server_use["input"]["url"], "https://example.com",
        "the reloaded block must keep its streamed input"
    );
    let result: Value =
        serde_json::from_str(passthroughs[1]).expect("reloaded block should be JSON");
    assert_eq!(result["type"], "web_search_tool_result");
}

/// A turn the server paused is not a finished answer. Reporting it as success
/// would hand the user a silently truncated response.
#[test]
fn cli_reports_a_paused_turn_as_incomplete() {
    let workspace = Workspace::create("paused");
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime should build");
    let server = runtime
        .block_on(MockAnthropicService::spawn())
        .expect("mock service should start");

    let output = run_claw(&workspace, &server.base_url(), "paused_turn", None);

    assert!(
        !output.status.success(),
        "a paused turn must not be reported as a successful completion"
    );
    let rendered = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        rendered.contains("pause_turn") || rendered.contains("incomplete"),
        "the user must be told the turn did not finish, got: {rendered}"
    );
}

/// Locate a session transcript beneath `root`. The sessions directory is derived
/// from a fingerprint of the workspace path, so it cannot be hard-coded here.
fn find_session_file(root: &Path) -> Option<PathBuf> {
    let entries = fs::read_dir(root).ok()?;
    let mut directories = Vec::new();
    for entry in entries.filter_map(Result::ok) {
        let path = entry.path();
        if path.is_dir() {
            directories.push(path);
        } else if path.extension().is_some_and(|ext| ext == "jsonl") {
            return Some(path);
        }
    }
    directories.iter().find_map(|dir| find_session_file(dir))
}
