// Phase 8: Compile .proto for gRPC server (only when "grpc" feature enabled).
// Without the feature, normal `cargo check` / `cargo build` succeeds even without protoc.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::var("CARGO_FEATURE_GRPC").is_ok() {
        tonic_build::configure()
            .build_server(true)
            .build_client(true)
            .compile(
                &["proto/vector_search.proto"],
                &["proto"],
            )?;
        println!("cargo:rerun-if-changed=proto/vector_search.proto");
    }
    Ok(())
}
