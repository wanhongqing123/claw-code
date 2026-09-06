//! Multi-AI Code 定制：与宿主 Electron 应用之间的结构化事件通道。
//!
//! 宿主在启动 claw 前开一个只监听 127.0.0.1 的 TCP 端口，并通过
//! `--multi-ai-code-im-ipc tcp://127.0.0.1:<port>?token=<token>` 把地址交给我们。
//! 之后双方用「一行一个 JSON」通信，每条都带 token。
//!
//! 出站（claw → 宿主）：
//!   {"token":"..","kind":"control_ready"}                        连上就发，见下
//!   {"token":"..","kind":"task_started","text":"<用户输入>","messageId":".."}
//!   {"token":"..","kind":"assistant_final","text":"<助手正文>","messageId":".."}
//!   {"token":"..","kind":"turn_error","text":"<错误信息>","messageId":".."}
//!   {"token":"..","kind":"control_result","requestId":"..","ok":true,"text":".."}
//!
//! 入站（宿主 → claw）：
//!   {"token":"..","kind":"ack","messageId":".."}
//!   {"token":"..","kind":"control","requestId":"..","command":"interrupt"}
//!
//! **`control_ready` 不能省**：宿主只把控制命令推给已经声明过 control_ready 的连接
//! （见宿主侧 writeControlPayload 只遍历 controlSockets）。不发就永远收不到控制命令，
//! 而且不会有任何报错——只是静默地什么都不发生。
//!
//! **为什么要 ack**：这条数据连接曾经出现过「半死」——socket 仍然可写、write 返回成功，
//! 但对端再也收不到，于是回传永久静默丢失、必须重启 AICLI 才恢复。所以每条带
//! messageId 的事件都要等回执；超时或写失败就重连并补发未确认的那几条。
//! 这与 codex/opencode 侧的做法一致，不要简化掉。

use std::collections::{HashSet, VecDeque};
use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// 未确认事件的重发上限。超过就丢弃最老的——回传是尽力而为，
/// 不能因为宿主长时间不在而把内存吃光。
const MAX_PENDING: usize = 8;
/// 等 ack 的时限；超过即认为连接半死。
const ACK_TIMEOUT: Duration = Duration::from_millis(1_500);
/// 轮询 ack 集合的间隔。
const ACK_POLL: Duration = Duration::from_millis(20);

#[derive(Debug, Clone)]
pub struct BridgeEndpoint {
    pub addr: String,
    pub token: String,
}

/// 解析 `tcp://127.0.0.1:1234?token=abc`。格式不认识就返回 None——
/// 桥是可选能力，解析失败必须让 claw 照常跑，不能因此启动不了。
pub fn parse_endpoint(raw: &str) -> Option<BridgeEndpoint> {
    let rest = raw.strip_prefix("tcp://")?;
    let (addr, query) = match rest.split_once('?') {
        Some((addr, query)) => (addr, query),
        None => (rest, ""),
    };
    if addr.is_empty() {
        return None;
    }
    let mut token = String::new();
    for pair in query.split('&') {
        if let Some(value) = pair.strip_prefix("token=") {
            token = percent_decode(value);
        }
    }
    if token.is_empty() {
        return None;
    }
    Some(BridgeEndpoint {
        addr: addr.to_string(),
        token,
    })
}

/// 只处理 %XX 与 '+'，够用且不引入依赖。
fn percent_decode(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'%' if i + 2 < bytes.len() => {
                let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).unwrap_or("");
                match u8::from_str_radix(hex, 16) {
                    Ok(byte) => {
                        out.push(byte);
                        i += 3;
                    }
                    Err(_) => {
                        out.push(bytes[i]);
                        i += 1;
                    }
                }
            }
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            byte => {
                out.push(byte);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// JSON 字符串转义。手写是为了不给这个 crate 增加序列化依赖——
/// 我们只发几个固定形状的对象，字段全部是字符串。
fn escape_json(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 8);
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// 从一行 JSON 里取某个字符串字段。只够解析我们自己定义的固定形状，
/// 不是通用 JSON 解析器——刻意不引依赖。
fn field(line: &str, name: &str) -> Option<String> {
    let key = format!("\"{name}\":\"");
    let start = line.find(&key)? + key.len();
    let rest = &line[start..];
    let mut out = String::new();
    let mut chars = rest.chars();
    while let Some(c) = chars.next() {
        match c {
            '"' => return Some(out),
            '\\' => match chars.next()? {
                'n' => out.push('\n'),
                'r' => out.push('\r'),
                't' => out.push('\t'),
                other => out.push(other),
            },
            other => out.push(other),
        }
    }
    None
}

#[derive(Debug, Clone)]
struct PendingEvent {
    message_id: String,
    line: String,
}

struct Shared {
    endpoint: BridgeEndpoint,
    stream: Option<TcpStream>,
    pending: VecDeque<PendingEvent>,
    acked: HashSet<String>,
    /// 当前这一轮的中断信号。每轮开始注册、结束清空。
    turn_signal: Option<runtime::HookAbortSignal>,
}

/// 事件通道。**所有失败都吞掉**：宿主没起、连接断了、对端不回执，
/// 都不能影响 claw 本身的交互——它首先是一个能独立使用的 CLI。
pub struct ImBridge {
    shared: Arc<Mutex<Shared>>,
    counter: AtomicU64,
}

impl ImBridge {
    pub fn connect(endpoint: BridgeEndpoint) -> Self {
        let stream = TcpStream::connect(&endpoint.addr).ok().inspect(|stream| {
            let _ = stream.set_nodelay(true);
        });
        let shared = Arc::new(Mutex::new(Shared {
            endpoint,
            stream,
            pending: VecDeque::new(),
            acked: HashSet::new(),
            turn_signal: None,
        }));
        let bridge = Self {
            shared: Arc::clone(&shared),
            counter: AtomicU64::new(0),
        };
        bridge.announce_and_spawn_reader();
        bridge
    }

    /// 发 control_ready 并起读线程。重连后要再走一遍。
    fn announce_and_spawn_reader(&self) {
        let Ok(mut shared) = self.shared.lock() else {
            return;
        };
        let Some(stream) = shared.stream.as_ref() else {
            return;
        };
        let Ok(mut write_half) = stream.try_clone() else {
            return;
        };
        let Ok(read_half) = stream.try_clone() else {
            return;
        };
        let ready = format!(
            "{{\"token\":\"{}\",\"kind\":\"control_ready\"}}\n",
            escape_json(&shared.endpoint.token)
        );
        if write_half.write_all(ready.as_bytes()).is_err() {
            shared.stream = None;
            return;
        }
        let _ = write_half.flush();
        drop(shared);

        let shared = Arc::clone(&self.shared);
        std::thread::spawn(move || {
            let mut reader = BufReader::new(read_half);
            loop {
                let mut line = String::new();
                match reader.read_line(&mut line) {
                    Ok(0) | Err(_) => return,
                    Ok(_) => handle_inbound(&shared, &line),
                }
            }
        });
    }

    fn next_message_id(&self) -> String {
        let n = self.counter.fetch_add(1, Ordering::Relaxed);
        format!("claw-{}-{}", std::process::id(), n)
    }

    /// 推一条事件并等待回执；没等到就重连补发。全部失败也只是静默返回。
    pub fn emit(&self, kind: &str, text: &str) {
        let message_id = self.next_message_id();
        let line = {
            let Ok(shared) = self.shared.lock() else {
                return;
            };
            format!(
                "{{\"token\":\"{}\",\"kind\":\"{}\",\"text\":\"{}\",\"messageId\":\"{}\"}}\n",
                escape_json(&shared.endpoint.token),
                escape_json(kind),
                escape_json(text),
                escape_json(&message_id)
            )
        };
        if let Ok(mut shared) = self.shared.lock() {
            shared.pending.push_back(PendingEvent {
                message_id,
                line,
            });
            while shared.pending.len() > MAX_PENDING {
                shared.pending.pop_front();
            }
        }
        self.flush();
    }

    fn flush(&self) {
        for attempt in 0..2 {
            {
                let Ok(mut shared) = self.shared.lock() else {
                    return;
                };
                if shared.stream.is_none() {
                    let addr = shared.endpoint.addr.clone();
                    shared.stream = TcpStream::connect(&addr).ok().inspect(|s| {
                        let _ = s.set_nodelay(true);
                    });
                    if shared.stream.is_some() {
                        drop(shared);
                        // 重连后必须重新声明 control_ready，否则宿主不会再给这条连接
                        // 推控制命令——它按连接维护 controlSockets。
                        self.announce_and_spawn_reader();
                    }
                }
            }
            let wrote = {
                let Ok(mut shared) = self.shared.lock() else {
                    return;
                };
                match shared.stream.as_ref().and_then(|s| s.try_clone().ok()) {
                    None => {
                        shared.stream = None;
                        false
                    }
                    Some(mut write_half) => {
                        let mut ok = true;
                        for event in shared.pending.iter() {
                            if write_half.write_all(event.line.as_bytes()).is_err() {
                                ok = false;
                                break;
                            }
                        }
                        ok && write_half.flush().is_ok()
                    }
                }
            };
            if !wrote {
                if let Ok(mut shared) = self.shared.lock() {
                    shared.stream = None;
                }
                continue;
            }
            // write 成功不代表对端收到——半死 socket 正是这样丢数据的。
            if self.wait_for_acks() {
                return;
            }
            if attempt == 0 {
                if let Ok(mut shared) = self.shared.lock() {
                    shared.stream = None;
                }
            }
        }
    }

    /// 等读线程把 ack 填进来。全部确认返回 true。
    fn wait_for_acks(&self) -> bool {
        let deadline = Instant::now() + ACK_TIMEOUT;
        loop {
            {
                let Ok(mut shared) = self.shared.lock() else {
                    return false;
                };
                let acked = std::mem::take(&mut shared.acked);
                shared.pending.retain(|event| !acked.contains(&event.message_id));
                // 还没对上的 ack 留着：事件可能刚写出去、ack 先到。
                shared.acked = acked;
                if shared.pending.is_empty() {
                    return true;
                }
            }
            if Instant::now() >= deadline {
                return false;
            }
            std::thread::sleep(ACK_POLL);
        }
    }

    fn send_line(&self, line: &str) {
        let Ok(mut shared) = self.shared.lock() else {
            return;
        };
        let Some(stream) = shared.stream.as_ref() else {
            return;
        };
        let Ok(mut write_half) = stream.try_clone() else {
            shared.stream = None;
            return;
        };
        if write_half.write_all(line.as_bytes()).is_err() {
            shared.stream = None;
            return;
        }
        let _ = write_half.flush();
    }
}

/// 处理一行入站消息。ack 记账，control 分派。
fn handle_inbound(shared: &Arc<Mutex<Shared>>, line: &str) {
    if line.contains("\"kind\":\"ack\"") {
        if let Some(id) = field(line, "messageId") {
            if let Ok(mut guard) = shared.lock() {
                guard.acked.insert(id);
            }
        }
        return;
    }
    if !line.contains("\"kind\":\"control\"") {
        return;
    }
    let request_id = field(line, "requestId");
    let command = field(line, "command").unwrap_or_default();

    let (ok, text) = match command.as_str() {
        "interrupt" => {
            let signal = shared.lock().ok().and_then(|guard| guard.turn_signal.clone());
            match signal {
                Some(signal) => {
                    signal.abort();
                    (true, "已请求中断：当前这轮读完即停".to_string())
                }
                // 没有正在跑的轮次时不算失败——IM 那端点「中断」时任务可能刚好结束了。
                None => (true, "当前没有正在进行的任务".to_string()),
            }
        }
        other => (
            false,
            format!("claw 尚未支持控制命令：{other}"),
        ),
    };

    let Some(request_id) = request_id else {
        return;
    };
    let (token, stream) = {
        let Ok(guard) = shared.lock() else {
            return;
        };
        (
            guard.endpoint.token.clone(),
            guard.stream.as_ref().and_then(|s| s.try_clone().ok()),
        )
    };
    let Some(mut stream) = stream else {
        return;
    };
    let reply = format!(
        "{{\"token\":\"{}\",\"kind\":\"control_result\",\"requestId\":\"{}\",\"ok\":{},\"text\":\"{}\"}}\n",
        escape_json(&token),
        escape_json(&request_id),
        if ok { "true" } else { "false" },
        escape_json(&text)
    );
    let _ = stream.write_all(reply.as_bytes());
    let _ = stream.flush();
}

/// 进程级单例。一个 claw 进程只对应一个宿主会话，所以用全局而不是把
/// 参数一路穿过 Command 枚举和 run_repl 签名——**刻意把 fork 补丁压到最小**，
/// 上游同步时要 rebase 的面积越小越好。
static BRIDGE: std::sync::OnceLock<Option<ImBridge>> = std::sync::OnceLock::new();

/// 由 `--multi-ai-code-im-ipc` 触发安装。解析不了就装成 None，claw 照常独立运行。
pub fn install(raw: &str) {
    let _ = BRIDGE.set(parse_endpoint(raw).map(ImBridge::connect));
}

/// 推一条事件；没有装桥时是空操作。
pub fn emit(kind: &str, text: &str) {
    if let Some(Some(bridge)) = BRIDGE.get() {
        bridge.emit(kind, text);
    }
}

/// 注册/清空当前轮次的中断信号。中断命令从读线程来，信号是 Arc<AtomicBool>，
/// 跨线程共享是安全的。
pub fn set_turn_signal(signal: Option<runtime::HookAbortSignal>) {
    if let Some(Some(bridge)) = BRIDGE.get() {
        if let Ok(mut shared) = bridge.shared.lock() {
            shared.turn_signal = signal;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_endpoint_with_token() {
        let parsed = parse_endpoint("tcp://127.0.0.1:5555?token=abc%2Fdef").expect("should parse");
        assert_eq!(parsed.addr, "127.0.0.1:5555");
        assert_eq!(parsed.token, "abc/def");
    }

    #[test]
    fn rejects_endpoint_without_token() {
        // 没有 token 就没法验证对端身份，必须当成「没有桥」而不是「连上去再说」。
        assert!(parse_endpoint("tcp://127.0.0.1:5555").is_none());
        assert!(parse_endpoint("http://127.0.0.1:5555?token=abc").is_none());
        assert!(parse_endpoint("garbage").is_none());
    }

    #[test]
    fn escapes_control_characters_in_payload() {
        // 助手正文里必然出现换行和引号；漏转义会把一行一 JSON 的协议打断。
        let escaped = escape_json("a\"b\\c\nd\te");
        assert_eq!(escaped, "a\\\"b\\\\c\\nd\\te");
    }

    #[test]
    fn reads_fields_and_unescapes_them() {
        let line = "{\"token\":\"t\",\"kind\":\"ack\",\"messageId\":\"claw-1-2\"}";
        assert_eq!(field(line, "messageId").as_deref(), Some("claw-1-2"));
        // 转义序列要还原，否则带引号或换行的字段会被截断。
        let escaped = "{\"command\":\"a\\\"b\",\"requestId\":\"r1\"}";
        assert_eq!(field(escaped, "command").as_deref(), Some("a\"b"));
        assert_eq!(field(escaped, "requestId").as_deref(), Some("r1"));
        assert!(field(line, "missing").is_none());
    }

    #[test]
    fn interrupt_aborts_the_registered_turn_signal() {
        let signal = runtime::HookAbortSignal::new();
        let shared = Arc::new(Mutex::new(Shared {
            endpoint: BridgeEndpoint {
                addr: "127.0.0.1:1".to_string(),
                token: "t".to_string(),
            },
            stream: None,
            pending: VecDeque::new(),
            acked: HashSet::new(),
            turn_signal: Some(signal.clone()),
        }));
        assert!(!signal.is_aborted());
        handle_inbound(
            &shared,
            "{\"token\":\"t\",\"kind\":\"control\",\"requestId\":\"r1\",\"command\":\"interrupt\"}",
        );
        assert!(signal.is_aborted(), "interrupt 必须真的把信号置位");
    }

    #[test]
    fn ack_frames_do_not_reach_the_control_path() {
        // 控制命令也带 messageId 形状的字段；把 ack 当控制命令会误触发中断。
        let signal = runtime::HookAbortSignal::new();
        let shared = Arc::new(Mutex::new(Shared {
            endpoint: BridgeEndpoint {
                addr: "127.0.0.1:1".to_string(),
                token: "t".to_string(),
            },
            stream: None,
            pending: VecDeque::new(),
            acked: HashSet::new(),
            turn_signal: Some(signal.clone()),
        }));
        handle_inbound(&shared, "{\"token\":\"t\",\"kind\":\"ack\",\"messageId\":\"m1\"}");
        assert!(!signal.is_aborted());
        assert!(shared.lock().unwrap().acked.contains("m1"));
    }
}
