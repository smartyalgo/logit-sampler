use std::{
    io::{Read, Write},
    net::{Shutdown, TcpListener, TcpStream},
    thread,
};

use crate::Args;

use rand::distr::{weighted::WeightedIndex, Distribution};
use tracing::{debug, error, info, warn};

use serde;

mod logit_manipulation;

#[derive(Debug, Clone)]
pub struct SamplingParams {
    temperature: Option<f32>,
    top_k: i64,
    top_p: f32,
    min_p: f32,
}

impl From<Args> for SamplingParams {
    fn from(args: Args) -> Self {
        Self {
            temperature: Some(args.temperature),
            top_k: args.top_k,
            top_p: args.top_p,
            min_p: args.min_p,
        }
    }
}

// This listens to sampler request, and routes them to the correct sampler.
#[derive(Debug)]
pub struct SamplerRouter {
    params: SamplingParams,
    inc_req_socket: TcpListener,
}

enum LlmConnectionState {
    Completed,
    NextToken,
}

pub struct Sampler {
    params: SamplingParams,
    llm_conn_stream: TcpStream,
    vocab: Vec<String>,
}

impl SamplerRouter {
    pub fn new(bind_address: &str, params: Option<SamplingParams>) -> Self {
        info!(bind_address, "Creating new Sampler. Params: {:?}", params);
        let tcp_listener = TcpListener::bind(bind_address).expect("Failed to bind to address");
        Self {
            inc_req_socket: tcp_listener,
            params: match params {
                Some(params) => params,
                None => SamplingParams {
                    temperature: None,
                    top_k: 1,
                    top_p: 1.0,
                    min_p: 0.0,
                },
            },
        }
    }

    pub fn run(&self) {
        for stream in self.inc_req_socket.incoming() {
            match stream {
                Ok(stream) => {
                    let params = self.params.clone();
                    thread::spawn(move || {
                        let mut sampler = Sampler::new(params, stream);
                        info!("New connection accepted. Handling handshake.");
                        if let Err(e) = sampler.handle_new_connection() {
                            error!(
                                ?e,
                                "Error at handshake - header exchange. Shutting down connection."
                            );
                            sampler.llm_conn_stream.shutdown(Shutdown::Both).unwrap();
                            return;
                        };

                        loop {
                            match sampler.handle_logits() {
                                Ok(LlmConnectionState::Completed) => {
                                    info!("Request completed, socket closed by client");
                                    break;
                                }
                                Ok(LlmConnectionState::NextToken) => {
                                    // NextToken is currently a passthru.
                                    // Uncomment the following to exit after processing the first token
                                    // debug!("Debug mode, processing only 1 token");
                                    // sampler.llm_conn_stream.shutdown(Shutdown::Both).unwrap();
                                    // break;
                                }
                                Err(e) => {
                                    error!(?e, "error while processing connection");
                                    break;
                                }
                            };
                        }
                    });
                }
                Err(e) => {
                    error!(?e, "Error accepting connection");
                    break;
                }
            }
        }
    }
}

impl Sampler {
    pub fn new(params: SamplingParams, llm_conn_stream: TcpStream) -> Self {
        Self {
            params,
            llm_conn_stream,
            vocab: Vec::new(),
        }
    }

    /// Handles a new connection by reading the handshake parameters and vocabulary tokens.
    ///
    /// # Arguments
    /// * `stream` - The TCP stream to read from
    ///
    /// # Returns
    /// * `Result<Vec<String>, Box<dyn std::error::Error>>` - The vocabulary tokens if successful, error otherwise
    ///
    /// # Protocol
    /// The handshake protocol expects:
    /// 1. A 13-byte header containing "HANDSHAKE" and the number of tokens
    /// 2. A sequence of null-terminated tokens
    /// 3. A final null byte
    ///
    /// # Errors
    /// Returns error if:
    /// * Failed to read from socket
    /// * Failed to parse handshake parameters
    /// * Client closes connection prematurely
    fn handle_new_connection(
        &mut self,
        // mut stream: &TcpStream,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // 13 is the length of "handshake" + i32 token length (4 bytes)
        let mut buf: [u8; 13] = [0; 13];
        let mut token_length: usize = 0;

        match self.llm_conn_stream.read_exact(&mut buf) {
            Ok(_) => match HandshakeParams::header_from_bytes(&buf[..13]) {
                Ok(length) => {
                    info!(token_length, "Successfully parsed handshake parameters");
                    token_length = length;
                }
                Err(e) => {
                    error!(?e, "Failed to parse handshake header parameters");
                    self.llm_conn_stream.shutdown(Shutdown::Both).unwrap();
                }
            },
            Err(e) => {
                error!(?e, "Error reading from socket");
                self.llm_conn_stream.shutdown(Shutdown::Both).unwrap();
            }
        }

        info!(token_length, "Received token length");

        let mut total_buff: Vec<u8> = Vec::new();
        // Swallow all bytes from socket until 2 null bytes are received. (This is a hack)
        // Then we take the vector of bytes, and deserialize the tokens.
        loop {
            let mut buf: [u8; 1024] = [0; 1024];
            match self.llm_conn_stream.read(&mut buf) {
                Ok(received_size) if received_size == 0 => {
                    warn!("Socket closed before handshake complete");
                    return Err("Socket closed by client".into());
                }
                Ok(received_size) => {
                    debug!(received_size, "Received data from client");

                    total_buff.extend_from_slice(&buf[..received_size]);

                    if total_buff.len() > 2
                        && total_buff[total_buff.len() - 1] == 0
                        && total_buff[total_buff.len() - 2] == 0
                    {
                        info!("Received null-terminated token");
                        break;
                    }
                }
                Err(e) => {
                    error!(?e, "Error reading from socket");
                    return Err(e.into());
                }
            }
        }

        debug!("Total buffer: {:?}", total_buff);
        let model_vocab = HandshakeParams::model_vocab_from_bytes(total_buff);
        debug!("token count: {}", model_vocab.len());

        if model_vocab.len() != token_length {
            error!(
                "Token count mismatch. Expected: {}, Extracted: {}",
                token_length,
                model_vocab.len()
            );
            self.llm_conn_stream
                .flush()
                .expect("Failed to flush stream");

            self.llm_conn_stream.shutdown(Shutdown::Both).unwrap();
            return Err("Token count mismatch".into());
        }

        self.vocab = model_vocab;
        let vocab_length: i32 = self.vocab.len() as i32;
        let mut response = Vec::new();
        response.extend_from_slice(&vocab_length.to_le_bytes());

        info!(
            "Sending handshake response: {:?}. Response length: {} bytes",
            response,
            response.len()
        );

        self.llm_conn_stream
            .write_all(&response)
            .expect("Failed to send handshake response[Token_count]");
        self.llm_conn_stream
            .flush()
            .expect("Failed to flush stream");

        info!("Handshake completed. Ready to receive logits.");
        Ok(())
    }

    fn handle_logits(&mut self) -> Result<LlmConnectionState, Box<dyn std::error::Error>> {
        let mut total_buff = Vec::new();
        let bytes_per_float = std::mem::size_of::<f32>();

        let vocab_in_bytes = self.vocab.len() * bytes_per_float + 6; // Prepended with LOGITS
        info!("Vocab in bytes: {}", vocab_in_bytes);
        let mut received_bytes = 0;

        // TODO: Extract L250 - 295 into logit serialization function
        loop {
            let mut buf: [u8; 2048] = [0; 2048];
            match self.llm_conn_stream.read(&mut buf) {
                Ok(received_size) if received_size == 0 => {
                    info!("Socket closed by client");
                    return Ok(LlmConnectionState::Completed);
                }
                Ok(received_size) => {
                    debug!(received_size, "Received logit data from client");
                    total_buff.extend_from_slice(&buf[..received_size]);

                    received_bytes += received_size;
                    debug!("Received {} bytes so far", received_bytes);

                    if received_bytes == vocab_in_bytes {
                        info!(
                            "Received all logits. Expected: {}. Received: {}",
                            vocab_in_bytes, received_bytes
                        );
                        break;
                    } else if received_bytes > vocab_in_bytes {
                        error!(
                            "Received more bytes than expected. Expected: {}. Received: {}",
                            vocab_in_bytes, received_bytes
                        );
                        return Err("Received more bytes than expected".into());
                    }
                }
                Err(e) => {
                    error!(?e, "Error reading from socket");
                }
            }
        }

        // Convert bytes to f32 values
        let mut logits: Vec<f32> = Vec::new();

        let logits_bytes = total_buff[6..].to_vec();

        for chunk in logits_bytes.chunks_exact(bytes_per_float) {
            if let Ok(bytes) = chunk.try_into() {
                let value = f32::from_le_bytes(bytes);
                logits.push(value);
            }
        }

        debug!("Logits: {:?}", logits);

        let logit_length: i32 = logits.len() as i32;
        let prob_logits = logit_manipulation::softmax(logits);
        let index: i32 = self.sample_logit(prob_logits) as i32;

        if let Some(token) = self.debug_token_from_index(index as usize) {
            info!("Sampled index {} maps to token: {:?}", index, token);
        }

        info!(
            "Logit length: {}. {:?}",
            logit_length,
            logit_length.to_le_bytes()
        );
        info!("Index: {}. {:?}", index, index.to_le_bytes());

        let mut response: Vec<u8> = Vec::new();
        response.extend_from_slice(&logit_length.to_le_bytes());
        response.extend_from_slice(&index.to_le_bytes());

        info!(
            "Sending logit response: {:?}\nResponse length: {} bytes",
            response,
            response.len()
        );
        self.llm_conn_stream
            .write_all(&response)
            .expect("Failed to send index");
        self.llm_conn_stream
            .flush()
            .expect("Failed to flush stream");

        Ok(LlmConnectionState::NextToken)
    }

    /// Debug helper that maps a token index back to its vocabulary entry.
    ///
    /// # Arguments
    /// * `index` - Token id, i.e. the position in the logit/vocab array.
    ///
    /// # Returns
    /// * `Some(&token)` for the vocab entry at `index`, or `None` if the index
    ///   is out of bounds.
    fn debug_token_from_index(&self, index: usize) -> Option<&str> {
        let token = self.vocab.get(index)?;
        debug!(index, token, "Resolved index to token");
        Some(token.as_str())
    }

    fn sample_logit(&self, mut logits: Vec<f32>) -> usize {
        if let Some(temp) = self.params.temperature {
            // info!("Temperature: {}", temp);
            // info!("Logits: {:?}", logits);

            if temp > 0.0 {
                logits = logits.iter().map(|logit| logit / temp).collect();
            }

            // info!("Logits after temperature: {:?}", logits);
        }

        let top_prob = if self.params.top_k > 0 {
            logit_manipulation::top_k_logits(&mut logits, self.params.top_k)
        } else {
            1.0
        };

        if self.params.top_p > 0.0 {
            logit_manipulation::top_p_logits(&mut logits, self.params.top_p);
        }

        if self.params.min_p > 0.0 {
            logit_manipulation::min_p_logits(&mut logits, self.params.min_p, top_prob);
        }

        let mut rng = rand::rng();

        let dist = WeightedIndex::new(logits).unwrap();
        let max_index = dist.sample(&mut rng);

        info!("Max index: {}", max_index);
        return max_index;
    }
}
#[derive(serde::Deserialize, serde::Serialize)]
pub struct HandshakeParams {
    pub handshake_param: Vec<String>,
}

impl HandshakeParams {
    fn header_from_bytes(bytes: &[u8]) -> Result<usize, Box<dyn std::error::Error>> {
        let expected_len = 13;
        if bytes.len() < expected_len {
            return Err(format!(
                "Insufficient bytes for handshake header. Received: {}. Expected: {}",
                bytes.len(),
                expected_len
            )
            .into());
        }

        let protocol_action = std::str::from_utf8(&bytes[..9])?;
        if protocol_action != "HANDSHAKE" {
            return Err("Invalid handshake magic string".into());
        }

        let token_length = &bytes[9..13];
        let token_length = i32::from_le_bytes(token_length.try_into().expect("Expect i32"));

        info!("token_length: {}", token_length);

        Ok(token_length as usize)
    }

    // Returns
    fn model_vocab_from_bytes(bytes: Vec<u8>) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut current_token = Vec::new();

        let mut processed_nul = false;

        info!("Bytes length: {}", bytes.len());
        for &byte in bytes.iter() {
            if byte == 0 {
                if !current_token.is_empty() {
                    debug!("Serializing token into vocab: {:?}", current_token);
                    let token = String::from_utf8(current_token);
                    match token {
                        Ok(token) => tokens.push(token),
                        Err(_e) => {
                            debug!("Encountered invalid utf8 byte sequence, pushing placeholder into vocab");
                            tokens.push("".to_string()); // We push a placeholder because these are invalid
                        }
                    }
                    current_token = Vec::new();
                } else if !processed_nul {
                    tokens.push("NUL".to_string());
                    processed_nul = true;
                }
            } else {
                current_token.push(byte);
            }
        }

        // info!("tokens: {:?}", tokens);
        tokens
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use std::io::Write;
    use std::net::{TcpListener, TcpStream};

    fn helper_generate_token_data() -> Vec<u8> {
        return vec![0; 256];
    }

    /// Builds a `Sampler` backed by a throwaway loopback connection so methods
    /// that only depend on `vocab` can be exercised in isolation.
    fn helper_sampler_with_vocab(vocab: Vec<String>) -> Sampler {
        let listener = TcpListener::bind("127.0.0.1:0").expect("Failed to bind listener");
        let addr = listener.local_addr().expect("Failed to read local addr");
        let stream = TcpStream::connect(addr).expect("Failed to connect to listener");

        let params = SamplingParams {
            temperature: None,
            top_k: 1,
            top_p: 1.0,
            min_p: 0.0,
        };

        let mut sampler = Sampler::new(params, stream);
        sampler.vocab = vocab;
        sampler
    }

    #[test]
    fn test_debug_token_from_index_returns_token() {
        let sampler = helper_sampler_with_vocab(vec![
            "hello".to_string(),
            "world".to_string(),
            "!".to_string(),
        ]);

        assert_eq!(sampler.debug_token_from_index(0), Some("hello"));
        assert_eq!(sampler.debug_token_from_index(1), Some("world"));
        assert_eq!(sampler.debug_token_from_index(2), Some("!"));
    }

    #[test]
    fn test_debug_token_from_index_out_of_bounds() {
        let sampler = helper_sampler_with_vocab(vec!["only".to_string()]);

        assert_eq!(sampler.debug_token_from_index(1), None);
        assert_eq!(sampler.debug_token_from_index(99), None);
    }

    // #[test]
    fn test_tcp_handshake() {
        // Attempt to connect to localhost on a specific port
        let mut stream = TcpStream::connect("127.0.0.1:8080").expect("Failed to connect to server");

        let protocol_action = "HANDSHAKE";
        let token_length: i32 = 256; // Why not u32?

        // These all go into HandshakeParams::to_bytes
        let mut data = Vec::new();
        data.extend_from_slice(protocol_action.as_bytes());
        data.extend_from_slice(&token_length.to_le_bytes());

        data.extend(helper_generate_token_data());

        stream
            .write_all(&data)
            .expect("Failed to send handshake parameter");

        stream.flush().expect("Failed to flush stream");
    }

    // TODO: Add test for malformed handshake
}
