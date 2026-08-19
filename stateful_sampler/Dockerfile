FROM rust:1-slim AS build
WORKDIR /app
COPY . .
RUN cargo build --release

FROM debian:bookworm-slim
COPY --from=build /app/target/release/stateful_sampler /usr/local/bin/stateful_sampler
EXPOSE 5146
ENTRYPOINT ["stateful_sampler"]
CMD ["0.0.0.0:5146"]
