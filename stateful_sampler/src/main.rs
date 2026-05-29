use clap::Parser;
use std::thread;
use tracing::debug;
use tracing_subscriber::fmt::format::FmtSpan;
use tracing_subscriber::EnvFilter;

mod sampler;

use sampler::SamplerRouter;

static DEFAULT_ADDRESS: &str = "0.0.0.0:5146";

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Server address in format host:port
    #[arg(default_value = DEFAULT_ADDRESS)]
    address: String,

    /// Temperature for sampling (optional)
    #[arg(short, long, default_value = "1.0")]
    temperature: f32,

    /// Top-k sampling parameter
    #[arg(long, default_value = "1")]
    top_k: i64,

    /// Top-p sampling parameter
    #[arg(long, default_value = "1.0")]
    top_p: f32,

    /// Min-p sampling parameter
    #[arg(long, default_value = "0.05")]
    min_p: f32,
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_file(true)
        .with_line_number(true)
        .with_thread_ids(false)
        .with_target(false)
        .with_span_events(FmtSpan::FULL)
        .with_level(true)
        .init();

    let args = Args::parse();

    debug!("Starting application");

    debug!("{:?}", args);

    // TODO: Can just pass 1 param
    let sampler = SamplerRouter::new(&args.address.clone(), Some(args.into()));
    let handle = thread::spawn(move || sampler.run());
    handle.join().unwrap();
}
