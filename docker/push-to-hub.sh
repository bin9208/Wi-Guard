#!/bin/bash
# ============================================
# RuView — Docker Hub 이미지 빌드 & 푸시
# by bin9208
# ============================================
#
# 사용법:
#   chmod +x docker/push-to-hub.sh
#   docker/push-to-hub.sh
#
# 사전 조건:
#   docker login
# ============================================

set -e

IMAGE_NAME="bin920832/ruview-nas"
TAG="latest"

echo "============================================"
echo "  RuView Docker Hub Push"
echo "  Image: ${IMAGE_NAME}:${TAG}"
echo "============================================"
echo ""

# 1. 빌드
echo "[1/3] Building Docker image..."
docker build \
    -f docker/Dockerfile.nas \
    -t "${IMAGE_NAME}:${TAG}" \
    .

echo ""

# 2. 태그
echo "[2/3] Tagging image..."
TIMESTAMP=$(date +%Y%m%d)
docker tag "${IMAGE_NAME}:${TAG}" "${IMAGE_NAME}:${TIMESTAMP}"
echo "  Tagged: ${IMAGE_NAME}:${TIMESTAMP}"

# 3. 푸시
echo "[3/3] Pushing to Docker Hub..."
docker push "${IMAGE_NAME}:${TAG}"
docker push "${IMAGE_NAME}:${TIMESTAMP}"

echo ""
echo "============================================"
echo "  ✅ Push 완료!"
echo ""
echo "  NAS에서 실행하려면:"
echo "    docker pull ${IMAGE_NAME}:${TAG}"
echo "    docker run -d --name ruview \\"
echo "      -p 8080:8080 -p 3000:3000 \\"
echo "      -p 3001:3001 -p 5005:5005/udp \\"
echo "      ${IMAGE_NAME}:${TAG}"
echo ""
echo "  또는 docker-compose.nas.yml 사용:"
echo "    docker-compose -f docker-compose.nas.yml up -d"
echo "============================================"
