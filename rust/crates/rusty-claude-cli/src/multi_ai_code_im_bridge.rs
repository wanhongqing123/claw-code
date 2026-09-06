//! Multi-AI Code 定制：把一轮对话的结构化事件推给宿主 Electron 应用。
//!
//! 宿主在启动 claw 前开一个只监听 127.0.0.1 的 TCP 端口，并通过
//! `--multi-ai-code-im-ipc tcp://127.0.0.1:<port>?token=<token>` 把地址交给我们。
//! 之后双方用「一行一个 JSON」通信，每条都带 token。
//!
//! 出站（claw → 宿主）：
//!   {"token":"..","kind":"task_started","text":"<用户输入>","messageId":".."}
//!   {"token":"..","kind":"assistant_final","text":"<助手正文>","messageId":".."}
//!   {"token":"..","kind":"turn_error","text":"<错误信息>","messageId":".."}
//!
//! 入站（宿主 → claw）：
//!   {"token":"..","kind":"ack","messageId":".."}
//!
//! **为什么要 ack**：这条数据连接曾经出现过「半死」——socket 仍然可写、write 返回成功，
//! 但对端再也收不到，于是回传永久静默丢失、必须重启 AICLI 才恢复。所以每条带
//! messageId 的事件都要等回执；超时或写失败就重连并补发未确认的那几条。
//! 这与 codex/opencode 侧的做法一致，不要简化掉。

use std::collections::VecDeque;
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
/// 重连退避。
const RECONNECT_BACKOFF: Duration = Duration::from_millis(300);

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
/// 我们只发三个固定形状的对象，字段全部是字符串。
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

#[derive(Debug, Clone)]
struct PendingEvent {
    message_id: String,
    line: String,
}

struct BridgeInner {
    endpoint: BridgeEndpoint,
    stream: Option<TcpStream>,
    pending: VecDeque<PendingEvent>,
}

/// 事件推送器。**所有失败都吞掉**：宿主没起、连接断了、对端不回执，
/// 都不能影响 claw 本身的交互——它首先是一个能独立使用的 CLI。
pub struct ImBridge {
    inner: Arc<Mutex<BridgeInner>>,
    counter: AtomicU64,
}

impl ImBridge {
    pub fn connect(endpoint: BridgeEndpoint) -> Self {
        let stream = TcpStream::connect(&endpoint.addr).ok().inspect(|stream| {
            let _ = stream.set_read_timeout(Some(ACK_TIMEOUT));
            let _ = stream.set_nodelay(true);
        });
        Self {
            inner: Arc::new(Mutex::new(BridgeInner {
                endpoint,
                stream,
                pending: VecDeque::new(),
            })),
            counter: AtomicU64::new(0),
        }
    }

    fn next_message_id(&self) -> String {
        let n = self.counter.fetch_add(1, Ordering::Relaxed);
        format!("claw-{}-{}", std::process::id(), n)
    }

    /// 推一条事件并等待回执；没等到就重连补发。全部失败也只是静默返回。
    pub fn emit(&self, kind: &str, text: &str) {
        let message_id = self.next_message_id();
        let Ok(mut inner) = self.inner.lock() else {
            return;
        };
        let line = format!(
            "{{\"token\":\"{}\",\"kind\":\"{}\",\"text\":\"{}\",\"messageId\":\"{}\"}}\n",
            escape_json(&inner.endpoint.token),
            escape_json(kind),
            escape_json(text),
            escape_json(&message_id)
        );
        inner.pending.push_back(PendingEvent {
            message_id: message_id.clone(),
            line,
        });
        while inner.pending.len() > MAX_PENDING {
            inner.pending.pop_front();
        }
        Self::flush(&mut inner);
    }

    fn flush(inner: &mut BridgeInner) {
        for _ in 0..2 {
            if inner.stream.is_none() {
                std::thread::sleep(RECONNECT_BACKOFF);
                inner.stream = TcpStream::connect(&inner.endpoint.addr).ok().inspect(|s| {
                    let _ = s.set_read_timeout(Some(ACK_TIMEOUT));
                    let _ = s.set_nodelay(true);
                });
            }
            let Some(stream) = inner.stream.as_ref() else {
                return;
            };
            let Ok(mut write_half) = stream.try_clone() else {
                inner.stream = None;
                continue;
            };

            let mut wrote_all = true;
            for event in inner.pending.iter() {
                if write_half.write_all(event.line.as_bytes()).is_err() {
                    wrote_all = false;
                    break;
                }
            }
            if !wrote_all || write_half.flush().is_err() {
                inner.stream = None;
                continue;
            }

            // 等回执。write 成功不代表对端收到——半死 socket 正是这样丢数据的。
            if Self::drain_acks(inner) {
                return;
            }
            inner.stream = None;
        }
    }

    /// 读 ack 并清掉已确认的事件。全部确认返回 true。
    fn drain_acks(inner: &mut BridgeInner) -> bool {
        let Some(stream) = inner.stream.as_ref() else {
            return false;
        };
        let Ok(read_half) = stream.try_clone() else {
            return false;
        };
        let mut reader = BufReader::new(read_half);
        let deadline = Instant::now() + ACK_TIMEOUT;
        while !inner.pending.is_empty() && Instant::now() < deadline {
            let mut line = String::new();
            match reader.read_line(&mut line) {
                Ok(0) => return false,
                Ok(_) => {
                    if let Some(id) = extract_ack_message_id(&line) {
                        inner.pending.retain(|event| event.message_id != id);
                    }
                }
                Err(_) => return false,
            }
        }
        inner.pending.is_empty()
    }
}

/// 从一行 JSON 里取 ack 的 messageId。只认 kind=ack，避免把别的消息当回执。
fn extract_ack_message_id(line: &str) -> Option<String> {
    if !line.contains("\"kind\":\"ack\"") {
        return None;
    }
    let key = "\"messageId\":\"";
    let start = line.find(key)? + key.len();
    let rest = &line[start..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
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
    fn reads_ack_message_id_only_from_ack_frames() {
        let ack = "{\"token\":\"t\",\"kind\":\"ack\",\"messageId\":\"claw-1-2\"}";
        assert_eq!(extract_ack_message_id(ack).as_deref(), Some("claw-1-2"));
        // 控制命令也带 messageId 形状的字段，不能被当成回执。
        let control = "{\"token\":\"t\",\"kind\":\"control\",\"messageId\":\"claw-1-2\"}";
        assert!(extract_ack_message_id(control).is_none());
    }
}
