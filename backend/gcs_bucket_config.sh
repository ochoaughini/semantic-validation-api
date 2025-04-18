#!/bin/zsh

# 4. Main update with validation
echo "⚙️ Aplicando novas políticas..." 2>&1 | tee -a storage_migration.log
typeset -A policies=(
    ["uniform-bucket-level-access"]="on"
    ["retention-period"]="30d"
    ["versioning"]="enabled"
)

for policy in ${(k)policies}; do
    if ! gcloud storage buckets update "gs://semval-bio-01_cloudbuild" "--$policy=${policies[$policy]}" &>> storage_migration.log; then
        echo "🔴 Falha ao aplicar $policy" >&2
        echo "ℹ️ Estado atual das políticas:"
        gsutil versioning get gs://semval-bio-01_cloudbuild
        gsutil uniformbucketlevelaccess get gs://semval-bio-01_cloudbuild
        exit 1
    fi
done

# 5. Secure structure reorganization
echo "📁 Reorganizando estrutura..." 2>&1 | tee -a storage_migration.log
if ! gsutil -m mv "gs://semval-bio-01_cloudbuild/source/*" "gs://semval-bio-01_cloudbuild/builds/backend/" &>> storage_migration.log; then
    echo "⚠️ Alguns arquivos não puderam ser movidos (continuando...)"
fi

# 6. Secure IAM configuration
echo "🔒 Aplicando políticas de acesso..." 2>&1 | tee -a storage_migration.log
CLOUD_BUILD_SA=$(gcloud projects get-iam-policy $(gcloud config get-value project) --filter="(bindings.role:roles/cloudbuild.builds.builder)" --format="value(bindings.members[0])")
if [ -z "$CLOUD_BUILD_SA" ]; then
    echo "⚠️ Service Account do Cloud Build não encontrada"
else
    gcloud storage buckets add-iam-policy-binding gs://semval-bio-01_cloudbuild \
        --member="$CLOUD_BUILD_SA" --role=roles/storage.objectAdmin &>> storage_migration.log
fi

echo "✅ Otimização concluída"
echo "🔍 Verificação final:"
gsutil ls -Lb gs://semval-bio-01_cloudbuild | grep -E 'Versioning|Retention|Lifecycle|IAM' | tee -a storage_migration.log

