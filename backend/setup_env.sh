#!/bin/zsh

# Configurações globais
LOG_FILE="docker_setup.log"
DOCKER_PREFS="$HOME/Library/Group Containers/group.com.docker/settings.json"

# Funções auxiliares
log() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

check_command() {
  if ! command -v $1 &> /dev/null; then
    log "ERRO: $1 não encontrado no PATH"
    return 1
  fi
  return 0
}

# 1. Verificação inicial do sistema
log "Iniciando configuração do ambiente Docker"
log "Verificando arquitetura do sistema..."

ARCH=$(uname -m)
case $ARCH in
  "arm64") 
    log "Arquitetura Apple Silicon (ARM) detectada"
    BREW_PATH="/opt/homebrew/bin/brew"
    ;;
  "x86_64") 
    log "Arquitetura Intel detectada"
    BREW_PATH="/usr/local/bin/brew"
    ;;
  *) 
    log "ERRO: Arquitetura não suportada: $ARCH"
    exit 1
    ;;
esac

# 2. Instalação do Homebrew (se necessário)
log "Verificando instalação do Homebrew..."
if ! check_command "brew"; then
  log "Instalando Homebrew..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  
  # Configurar Homebrew no PATH
  echo 'eval "$('$BREW_PATH' shellenv)"' >> ~/.zshrc
  eval "$($BREW_PATH shellenv)"
fi

# 3. Instalação do Docker Desktop via Homebrew
log "Verificando instalação do Docker..."
if ! check_command "docker"; then
  log "Instalando Docker Desktop via Homebrew..."
  brew install --cask docker

  log "Abrindo Docker Desktop para configuração inicial..."
  open -a Docker

  log "Aguardando inicialização do Docker Desktop..."
  while ! docker system info &>/dev/null; do
    sleep 5
    log "Aguardando Docker daemon..."
  done
fi

# 4. Configuração do ambiente Docker
log "Configurando ambiente Docker..."

# Alocação de recursos (CPU/Memória)
log "Configurando recursos do Docker..."
cat << EOF > ~/docker_settings.json
{
  "cpus": $(sysctl -n hw.ncpu),
  "memory": $(($(sysctl -n hw.memsize) / 1024 / 1024 / 2)),
  "diskSize": 128,
  "vmType": "qemu",
  "useVirtualizationFramework": true
}
EOF

# Configuração do daemon Docker
log "Criando configuração do daemon Docker..."
DOCKER_CONFIG_DIR="$HOME/.docker"
mkdir -p "$DOCKER_CONFIG_DIR"

cat << EOF > "$DOCKER_CONFIG_DIR/daemon.json"
{
  "features": {
    "buildkit": true
  },
  "experimental": false,
  "debug": true,
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  }
}
EOF

# 5. Utilitários e verificação final
log "Instalando utilitários adicionais..."
brew install docker-compose docker-credential-helper

log "Verificando instalação..."
check_command "docker" && docker --version
check_command "docker-compose" && docker-compose --version

log "Realizando teste de funcionamento..."
if docker run --rm hello-world; then
  log "Docker configurado com sucesso!"
else
  log "ERRO: Problema na configuração do Docker"
  exit 1
fi

log "Configuração concluída! Detalhes no arquivo $LOG_FILE"

