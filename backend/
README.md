# Configuração do Ambiente Docker

## Pré-requisitos
- macOS 12+ (Monterey) ou superior
- Terminal com zsh
- Acesso administrativo local

## Instalação Automática
```bash
chmod +x setup_env.sh
./setup_env.sh
```

## Configuração Manual (alternativa)
1. Instalar Docker Desktop: https://docs.docker.com/desktop/install/mac-install/
2. Configurar recursos no Docker Desktop (CPU: 50%, RAM: 50%)
3. Habilitar Kubernetes (opcional)

## Verificação
```bash
docker --version
docker-compose --version
docker run --rm hello-world
```

## Solução de Problemas
Consulte o arquivo `docker_setup.log` para detalhes da instalação.

