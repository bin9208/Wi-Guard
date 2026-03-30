# 1단계: 빌드 스테이지 (Debian Bookworm 기반으로 GLIBC 버전 일치)
FROM rust:1-bookworm AS builder

# 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# CPU 전용 Libtorch (cxx11 ABI) 다운로드 및 설정
# tch-rs 0.18.1 버전에 맞춰 Libtorch 2.5.1 사용
WORKDIR /opt
RUN wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcpu.zip -O libtorch.zip \
    && unzip libtorch.zip \
    && rm libtorch.zip

ENV LIBTORCH=/opt/libtorch
ENV LD_LIBRARY_PATH=/opt/libtorch/lib

# 소스 코드 복사 및 빌드
WORKDIR /usr/src/wi-guard
COPY . .

# sensing-server 크레이트만 릴리스 모드로 빌드
RUN cargo build --release -p wifi-densepose-sensing-server

# 2단계: 런타임 스테이지
FROM debian:bookworm-slim

# 최소 런타임 라이브러리 설치
RUN apt-get update && apt-get install -y \
    libssl3 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 빌드된 바이너리 복사
COPY --from=builder /usr/src/wi-guard/target/release/sensing-server /app/

# Libtorch 라이브러리 복사
COPY --from=builder /opt/libtorch/lib/*.so* /usr/local/lib/

# 모델 체크포인트 및 설정 파일 내장 (사용자 요청사항)
COPY checkpoint /app/checkpoint

# UI 정적 파일 복사
COPY ui /app/ui

# 라이브러리 경로 설정
ENV LD_LIBRARY_PATH=/usr/local/lib

# 포트 설정: TCP 8080 (Web UI), UDP 5005 (ESP32 CSI)
EXPOSE 8080/tcp
EXPOSE 5005/udp

# 실행 명령
ENTRYPOINT ["/app/sensing-server", "--ui-path", "/app/ui"]
CMD ["--model-checkpoint", "/app/checkpoint/best_epoch0036_pck0.8908.safetensors", "--source", "esp32"]
